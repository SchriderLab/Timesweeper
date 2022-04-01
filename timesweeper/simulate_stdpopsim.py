import argparse as ap
import logging
import multiprocessing as mp
import os, shutil, sys
import random
import re
import subprocess
from itertools import cycle

import numpy as np
import pandas as pd

from timesweeper import read_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel("INFO")

seed = 42
random.seed(seed)
np.random.seed(seed)


def get_slim_code(slim_file):
    with open(slim_file, "r") as infile:
        raw_lines = [line.replace("\n", "") for line in infile.readlines()]

    return raw_lines


# fmt: off
def get_slim_info(slim_lines):
    def get_ints(lines, query):
        """Splits a line by regex split and grabs all digits. Returns as list."""
        # Uses max to handle max_years_b0, but might need to check if that causes other parsing issues
        return [
            int(s)
            for s in re.split("\W+", [i for i in lines if query in i][0])
            if s.isdigit()
        ]

    Q = get_ints(slim_lines, """defineConstant("Q",""")[0]
    gen_time = get_ints(slim_lines, """defineConstant("generation_time",""")[0]
    burn_time_mult = get_ints(slim_lines, """defineConstant("burn_in",""")[0]
    max_years_b0 = max(get_ints(slim_lines, """defineConstant("_T",""")) #Number of years BP to sim after burn in
    physLen = get_ints(slim_lines, """defineConstant("chromosome_length",""")[0]

    pop_sizes_line = [i for i in slim_lines if "_N" in i][0]
    pop_sizes_start = slim_lines.index(pop_sizes_line) + 2
    # The 4,3 will need to be more flexible
    # End of definition idx, will be end of sampling ep array
    pop_sizes_end = (slim_lines[pop_sizes_start:].index('    defineConstant("num_epochs", length(_T));') - 2 + pop_sizes_start)  

    sizes = []
    for i in slim_lines[pop_sizes_start:pop_sizes_end]:
        sizes.append([int(s) for s in re.split("\W+", i) if s.isdigit()])

    first_biggest = max([i[0] for i in sizes])
    burn_in_gens = first_biggest * burn_time_mult

    return Q, gen_time, max_years_b0, round(burn_in_gens), physLen


def inject_constants(raw_lines, sweep, selCoeff, sel_gen, end_gen, mut_rate, dumpfile):
    """Adds in sweep type, selection coefficient, and some other details."""
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tdefineConstant('sweep', '{sweep}');",
    )
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tdefineConstant('selCoeff', Q * {selCoeff});",
    )
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tinitializeMutationType('m2', 0.5, 'f', selCoeff);",
    )
    
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tdefineConstant('dumpFile', '{dumpfile}');"
    )

    raw_lines.insert(
        raw_lines.index("""    sim.registerLateEvent(NULL, "{dbg(self.source); end();}", G_end, G_end);"""),
        f"""    sim.registerLateEvent(NULL, "{{dbg(self.source); checkOnSweep(); }}", {sel_gen}, {end_gen});"""
    )

    raw_lines[raw_lines.index('    defineConstant("mutation_rate", Q * 0);')] = f'    defineConstant("mutation_rate", Q * {mut_rate});'
    
    return raw_lines


def sanitize_slim(raw_lines):
    """Removes sections of code inserted by stdpopsim that aren't needed."""
    raw_lines.pop(raw_lines.index("    sim.treeSeqOutput(trees_file);"))
    raw_lines.pop(raw_lines.index("    initializeTreeSeq();"))
    raw_lines.pop(
        raw_lines.index("""            "inds=p"+pop+".sampleIndividuals("+n+"); " +""")
    )
    return raw_lines


def inject_sampling(raw_lines, pop, samp_counts, gens, outfile_path):
    """Injects the actual sampling block that outputs to VCF."""
    samp_eps_line = [i for i in raw_lines if "sampling_episodes" in i][0]
    # Start of sampling_episodes constant idx
    samp_eps_start = raw_lines.index(samp_eps_line)  
    # End of definition idx, will be end of sampling ep array
    samp_eps_end = (raw_lines[samp_eps_start:].index("    ), c(3, 3)));") + samp_eps_start) 

    new_lines = []
    new_lines.extend(raw_lines[samp_eps_start : samp_eps_start + 1])
    last_line_id = len(samp_counts)
    for samps, gen, idx in zip(samp_counts, gens, range(last_line_id)):
        if idx < last_line_id - 1:
            new_lines.append("\t\t" + f"c({pop[1:]}, {samps}, {gen}),")
        else:
            new_lines.append("\t\t" + f"c({pop[1:]}, {samps}, {gen})")
    new_lines.append("\t" + f"), c(3, {len(gens)})));")

    for line in raw_lines:
        if "treeSeqRememberIndividuals" in line:
            raw_lines[
                raw_lines.index(line)
            ] = f"""\t\t\t"{pop}.outputVCFSample("+n+", replace=T, filePath='{outfile_path}', append=T);}}","""

    finished_lines = raw_lines[:samp_eps_start]
    finished_lines.extend(new_lines)
    finished_lines.extend(raw_lines[samp_eps_end + 1 :])

    return finished_lines


def make_sel_blocks(sweep, sel_gen, pop, dumpFileName):
    if sweep == "soft":
        restart_gen = sel_gen - 500
    else:
        restart_gen = sel_gen

    intro_block = f"""
    \n{restart_gen} late(){{
        // save the state of the simulation 
        print("SAVING TO " + "{dumpFileName}" + " at generation " + sim.generation);
        sim.outputFull("{dumpFileName}");

        if (sweep == "hard")
        {{    
            // introduce the sweep mutation
            target = sample({pop}.genomes, 1);
            target.addNewDrawnMutation(m2, asInteger(chromosome_length/2));
        }}
    }}

    """

    soft_block = f"""
    \n{sel_gen} late(){{
        if (sweep == "soft")
        {{
            muts = sim.mutationsOfType(m1);
            if (size(muts))
            {{
                mut = NULL;
                minDist = chromosome_length+1;
                for (m in muts)
                {{
                    freq = sim.mutationFrequencies({pop}, m);
                    if (freq > 0)
                    {{
                        dist = abs(m.position-asInteger(chromosome_length/2));
                        if (dist < minDist)
                        {{
                            minDist = dist;
                            mut = m;
                        }}
                    }}
                }}

                print("Chosen mut:" + mut.id);
                mut.setMutationType(m2);
                mut.setSelectionCoeff(selCoeff);
                
                print("Chose polymorphism at position " + mut.position + " and frequency " + sim.mutationFrequencies({pop}, mut) + " to become beneficial at generation " + sim.generation);

            }}
            else
            {{
                print("Failed to switch from neutral to beneficial at gen " + sim.generation);
            }}
        }}
    }}
    """

    check_block = f"""
    \nfunction (void)checkOnSweep(void) {{
        m1.convertToSubstitution = F;
        m2.convertToSubstitution = F;

        if (sweep == "hard" | (sweep == "soft"))
        {{
            fixed = (sum(sim.substitutions.mutationType == m2) == 1);
            if (fixed)
            {{
                print("FIXED in pop 1 at gen " + sim.generation);
                sim.deregisterScriptBlock(self);
            }}
            else
            {{
                muts = sim.mutationsOfType(m2);
                if (size(muts) == 0)
                {{
                    print("LOST at gen " + sim.generation + " - RESTARTING");
                    // Reload
                    sim.readFromPopulationFile("{dumpFileName}");
                
                    // Start a newly seeded run
                    setSeed(rdunif(1, 0, asInteger(2^32) - 1));
                
                    if (sweep == "hard")
                    {{
                        // re-introduce the sweep mutation
                        target = sample({pop}.genomes, 1);
                        print("RE-INTRODUCED MUTATION at gen " + sim.generation); //" with 2Ns = " + 2*subpopSize*selCoeff);
                        target.addNewDrawnMutation(m2, asInteger(chromosome_length/2));
                    }}
                }}
            }}
        }}
    }}

    """

    all_blocks = []
    for i in intro_block, soft_block, check_block:
        all_blocks.extend(i.split("\n"))

    return all_blocks
# fmt: on


def randomize_selCoeff(bounds=[0.005, 0.5]):
    """Draws selection coefficient from log normal dist to vary selection strength."""
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    log_low = np.math.log10(lower_bound)
    log_upper = np.math.log10(upper_bound)
    rand_log = np.random.uniform(log_low, log_upper, 1)

    return 10 ** rand_log[0]


def randomize_selTime(sel_time, stddev):
    """Draws from uniform dist to vary the time selection is induced before sampling in gens."""
    return int(np.random.uniform(sel_time - stddev, sel_time + stddev, 1)[0])


def write_slim(finished_lines, slim_file, rep, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    filename = os.path.basename(slim_file).split(".")[0]
    new_file_name = os.path.join(work_dir, f"modded.{rep}.{filename}.slim")
    with open(new_file_name, "w") as outfile:
        for line in finished_lines:
            outfile.write(line + "\n")

    return new_file_name


def run_slim(slimfile, slim_path):
    cmd = f"{slim_path} {slimfile}"
    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        logger.error(e.output)

    sys.stdout.flush()
    sys.stderr.flush()


def get_ua():
    agp = ap.ArgumentParser(
        description="Injects time-series sampling into stdpopsim SLiM output script."
    )
    agp.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        dest="verbose",
        help="Print verbose logging during process.",
    )
    agp.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count() - 1,
        dest="threads",
        help="Number of processes to parallelize across.",
    )
    agp.add_argument(
        "--rep-range",
        required=False,
        dest="rep_range",
        nargs=2,
        help="<start, stop>. If used, only range(start, stop) will be simulated for reps. \
            This is to allow for easy SLURM parallel simulations.",
    )
    subparsers = agp.add_subparsers(dest="config_format")
    subparsers.required = True
    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "-i",
        "--slim-file",
        required=True,
        type=str,
        help="SLiM Script output by stdpopsim to add time-series sampling to.",
        dest="slim_file",
    )
    cli_parser.add_argument(
        "--reps",
        required=True,
        type=int,
        help="Number of replicate simulations to run. If using rep_range can just fill with random int.",
        dest="reps",
    )
    cli_parser.add_argument(
        "--pop",
        required=False,
        type=str,
        default="p2",
        dest="pop",
        help="Label of population to sample from, will be defined in SLiM Script. Defaults to p2.",
    )
    cli_parser.add_argument(
        "--sample_sizes",
        required=True,
        type=int,
        nargs="+",
        dest="sample_sizes",
        help="Number of individuals to sample without replacement at each sampling point. Will be multiplied by ploidy to sample chroms from slim. Must match the number of entries in the -y flag.",
    )
    cli_parser.add_argument(
        "--years-sampled",
        required=True,
        type=int,
        nargs="+",
        dest="years_sampled",
        help="Years BP (before 1950) that samples are estimated to be from. Must match the number of entries in the -n flag.",
    )
    cli_parser.add_argument(
        "--selection-generation",
        required=False,
        type=int,
        default=200,
        dest="sel_gen",
        help="Number of gens before first sampling to introduce selection in population. Defaults to 200.",
    )
    cli_parser.add_argument(
        "--selection-coeff-bounds",
        required=False,
        type=float,
        default=[0.005, 0.5],
        dest="sel_coeff_bounds",
        action="append",
        nargs=2,
        help="Bounds of log-uniform distribution for pulling selection coefficient of mutation being introduced. Defaults to [0.005, 0.5]",
    )
    cli_parser.add_argument(
        "--mut-rate",
        required=False,
        type=str,
        default="1.29e-8",
        dest="mut_rate",
        help="Mutation rate for mutations not being tracked for sweep detection. Defaults to 1.29e-8 as defined in stdpopsim for OoA model.",
    )
    cli_parser.add_argument(
        "--work-dir",
        required=False,
        type=str,
        default="./ts_experiment",
        dest="work_dir",
        help="Directory to start workflow in, subdirs will be created to write simulation files to. Will be used in downstream processing as well.",
    )
    cli_parser.add_argument(
        "--slim-path",
        required=False,
        type=str,
        default="../SLiM/build/slim",
        dest="slim_path",
        help="Path to SLiM executable.",
    )

    ua = agp.parse_args()

    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        (
            slim_file,
            reps,
            pop,
            sample_sizes,
            years_sampled,
            sel_gen,
            sel_coeff_bounds,
            mut_rate,
            work_dir,
            slim_path,
        ) = (
            yaml_data["slimfile"],
            yaml_data["reps"],
            yaml_data["pop"],
            yaml_data["sample sizes"],
            yaml_data["years sampled"],
            yaml_data["selection gen"],
            yaml_data["selection coeff bounds"],
            yaml_data["mut rate"],
            yaml_data["work dir"],
            yaml_data["slim path"],
        )
    elif ua.config_format == "cli":
        (
            slim_file,
            reps,
            pop,
            sample_sizes,
            years_sampled,
            sel_gen,
            sel_coeff_bounds,
            mut_rate,
            work_dir,
            slim_path,
        ) = (
            ua.slim_file,
            ua.reps,
            ua.pop,
            ua.sample_sizes,
            ua.years_sampled,
            ua.sel_gen,
            ua.sel_coeff_bounds,
            ua.mut_rate,
            ua.work_dir,
            ua.slim_path,
        )

    if ua.verbose:
        logger.info(f"Number of sample sizes: {len(sample_sizes)}")
        logger.info(f"Number of years to sample from: {len(years_sampled)}")

    if len(sample_sizes) != len(years_sampled):
        logger.error(
            "Number of args supplied for generations and sample sizes don't match up. Please double check your values."
        )
        raise ValueError()

    if years_sampled[0] < years_sampled[-1]:
        logger.warning(
            "Gens were not supplied in earliest to latest order, sorting and flipping."
        )
        years_sampled = sorted(years_sampled)[::-1]
        sample_sizes = sorted(sample_sizes)[::-1]
    else:
        years_sampled = years_sampled
        sample_sizes = sample_sizes

    return (
        ua.verbose,
        ua.threads,
        ua.rep_range,
        work_dir,
        slim_file,
        reps,
        pop,
        sample_sizes,
        years_sampled,
        sel_gen,
        sel_coeff_bounds,
        mut_rate,
        slim_path,
    )


def main():
    (
        verbose,
        threads,
        rep_range,
        work_dir,
        slim_file,
        reps,
        pop,
        sample_sizes,
        years_sampled,
        sel_gen,
        sel_coeff_bounds,
        mut_rate,
        slim_path,
    ) = get_ua()

    work_dir = work_dir
    vcf_dir = f"{work_dir}/vcfs"
    dumpfile_dir = f"{work_dir}/dumpfiles"
    script_dir = f"{work_dir}/scripts"

    sweeps = ["neut", "hard", "soft"]

    for i in [vcf_dir, dumpfile_dir, script_dir]:
        for sweep in sweeps:
            os.makedirs(f"{i}/{sweep}", exist_ok=True)

    # Inject info into SLiM script and then simulate, store params for reproducibility
    sim_params = []
    script_list = []

    if rep_range:  # Take priority
        replist = range(int(rep_range[0]), int(rep_range[1]) + 1)
    else:
        replist = range(reps)

    for rep in replist:
        for sweep in sweeps:

            # Info scraping and calculations
            raw_lines = get_slim_code(slim_file)
            raw_lines = sanitize_slim(raw_lines)

            Q, gen_time, max_years_b0, burn_in_gens, physLen = get_slim_info(raw_lines)

            # Pull from variable time of selection before sampling to make more robust
            rand_sel_gen = randomize_selTime(sel_gen, 50 / Q)

            burn_in_gens = int(round(burn_in_gens / Q))

            # Convert from earliest year from bp to gens
            end_gen = int(round((max_years_b0) / gen_time / Q))

            # Written out each step for clarity, hard to keep track of otherwise
            # Find the earliest time before present, convert to useable times
            furthest_from_pres = max(years_sampled)
            abs_year_beg = max_years_b0 - furthest_from_pres

            sel_gen_time = (
                int(((abs_year_beg / gen_time) - rand_sel_gen) / Q)
            ) + burn_in_gens
            sel_coeff = randomize_selCoeff(sel_coeff_bounds)
            sel_coeff = sel_coeff * Q

            # Logger vars
            if verbose:
                logger.info("Timesweeper SLiM Injection")
                logger.info(f"Q Scaling Value: {Q}")
                logger.info(f"Gen Time: {gen_time}")
                logger.info(f"Simulated Chrom Length: {physLen}")
                logger.info(f"Burn in years: {burn_in_gens * gen_time}")
                logger.info(f"Burn in gens: {burn_in_gens}")
                logger.info(f"Number Years Simulated (post-burn): {max_years_b0}")
                logger.info(f"Number Gens Simulated (post-burn): {end_gen}")
                logger.info(
                    f"Number Years Simulated (inc. burn): {max_years_b0 + (burn_in_gens * gen_time)}"
                )
                logger.info(
                    f"Number gens simulated (inc. burn): {end_gen + burn_in_gens}"
                )
                logger.info(f"Selection type: {sweep}")
                logger.info(f"Selection start gen: {sel_gen_time}")
                logger.info(f"Number of timepoints: {len(years_sampled)}")
                logger.info(
                    f"""Sample sizes (individuals): 
                    {" ".join([str(i) for i in sample_sizes])}"""
                )
                logger.info(
                    f"Years before present (1950) sampled: {' '.join([str(i) for i in years_sampled])}"
                )
                logger.info(
                    f"Gens before present (1950) sampled: {' '.join([str(int(i / gen_time / Q)) for i in years_sampled])}",
                )

            sim_params.append(
                (
                    sweep,
                    rep,
                    Q,
                    gen_time,
                    physLen,
                    burn_in_gens,
                    max_years_b0 + (burn_in_gens * gen_time),
                    sel_coeff,
                    sel_gen_time,
                    years_sampled,
                    sample_sizes,
                )
            )

            dumpfile = f"{dumpfile_dir}/{sweep}/{rep}.dump"

            # Injection
            prepped_lines = inject_constants(
                raw_lines,
                sweep,
                sel_coeff,
                sel_gen_time,
                end_gen + burn_in_gens,
                mut_rate,
                dumpfile,
            )

            sampling_lines = inject_sampling(
                prepped_lines,
                pop,
                sample_sizes,
                years_sampled,
                f"{vcf_dir}/{sweep}/{rep}.multivcf",
            )

            selection_lines = make_sel_blocks(sweep, sel_gen_time, pop, dumpfile)
            finished_lines = []
            finished_lines.extend(sampling_lines)
            finished_lines.extend(selection_lines)

            script_path = write_slim(
                finished_lines, slim_file, rep, f"{script_dir}/{sweep}"
            )
            script_list.append(script_path)

    pool = mp.Pool(processes=threads)
    pool.starmap(run_slim, zip(script_list, cycle([slim_path])))

    # Cleanup
    for rep in replist:
        for sweep in sweeps:
            dumpFile = f"{dumpfile_dir}/{sweep}/{rep}.dump"
            os.remove(dumpFile)

    for scriptfile in script_list:
        os.remove(scriptfile)

    # Save params
    if not os.path.exists(f"{work_dir}/params"):
        os.makedirs(f"{work_dir}/params")
    params_df = pd.DataFrame(
        sim_params,
        columns=[
            "sweep",
            "rep",
            "Q",
            "gen_time",
            "chrom_len",
            "burn_in_gens",
            "num_gens_simmed",
            "sel_coeff",
            "sel_gen",
            "years_bp_sampled",
            "samp_sizes",
        ],
    )
    params_df.to_csv(
        f"{work_dir}/params/sim_{'_'.join(rep_range)}_params.tsv", sep="\t", index=False
    )

    logger.info(f"Simulations finished, parameters saved to {work_dir}/sim_params.csv.")


if __name__ == "__main__":
    main()
