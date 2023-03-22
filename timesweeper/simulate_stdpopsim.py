import argparse as ap
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
from glob import glob
from itertools import cycle

import numpy as np
import pandas as pd

from timesweeper import simulate_custom as sc
from timesweeper.utils.gen_utils import read_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel("INFO")


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


def inject_constants(raw_lines, sweep, recombRate, selCoeff, sel_gen, end_gen, mut_rate, dumpfile):
    """Adds in sweep type, selection coefficient, and some other details."""
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tdefineConstant('sweep', '{sweep}');",
    )
    raw_lines.insert(
        raw_lines.index("    initializeMutationRate(mutation_rate);"),
        f"\tdefineConstant('selCoeff', Q * {selCoeff});",
    )
    recomb_idx = raw_lines.index("    _recombination_rates = c(")
    raw_lines[recomb_idx] = f"    _recombination_rates = c({recombRate});"
    raw_lines[recomb_idx+1] = "\n"

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
            ] = f"""\t\t\t"{pop}.outputVCFSample("+n+", replace=F, filePath='{outfile_path}', append=T);}}","""

    finished_lines = raw_lines[:samp_eps_start]
    finished_lines.extend(new_lines)
    finished_lines.extend(raw_lines[samp_eps_end + 1 :])

    return finished_lines


def make_sel_blocks(sweep, sel_gen, pop, dumpFileName):
    if sweep == "ssv":
        restart_gen = sel_gen - 500
    else:
        restart_gen = sel_gen

    intro_block = f"""
    \n{restart_gen} late(){{
        // save the state of the simulation 
        print("SAVING TO " + "{dumpFileName}" + " at generation " + sim.generation);
        sim.outputFull("{dumpFileName}");

        if (sweep == "sdn")
        {{    
            // introduce the sweep mutation
            target = sample({pop}.genomes, 1);
            target.addNewDrawnMutation(m2, asInteger(chromosome_length/2));
        }}
        m1.convertToSubstitution = F;
        m2.convertToSubstitution = F;

    }}

    """

    ssv_block = f"""
    \n{sel_gen} late(){{
        if (sweep == "ssv")
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
        m1.convertToSubstitution = F;
        m2.convertToSubstitution = F;

    }}
    """

    check_block = f"""
    \nfunction (void)checkOnSweep(void) {{

        if (sweep == "sdn" | (sweep == "ssv"))
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
                
                    if (sweep == "sdn")
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
    for i in intro_block, ssv_block, check_block:
        all_blocks.extend(i.split("\n"))

    return all_blocks
# fmt: on


def randomize_selCoeff_loguni(lower_bound=0.005, upper_bound=0.5):
    """Draws selection coefficient from log uniform dist to vary selection strength."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )
    log_low = np.math.log10(lower_bound)
    log_upper = np.math.log10(upper_bound)
    rand_log = rng.uniform(log_low, log_upper, 1)

    return 10 ** rand_log[0]


def randomize_selCoeff_uni(lower_bound=0.00025, upper_bound=0.25):
    """Draws selection coefficient from log uniform dist to vary selection strength."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )

    return rng.uniform(lower_bound, upper_bound, 1)[0]


def randomize_recombRate(lower_bound=0, upper_bound=2e-8):
    """Draws selection coefficient from log normal dist to vary selection strength."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )
    recomb_rate = rng.uniform(lower_bound, upper_bound, 1)

    return recomb_rate[0]


def randomize_selTime(sel_time, stddev):
    """Draws from uniform dist to vary the time selection is induced before sampling in gens."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )
    return int(rng.uniform(sel_time - stddev, sel_time + stddev, 1)[0])


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


def process_vcfs(input_vcf, num_tps):
    try:
        # Split into multiples after SLiM just concats to same file
        raw_lines = sc.read_multivcf(input_vcf)
        split_lines = sc.split_multivcf(raw_lines, "##fileformat=VCFv4.2")
        if len(split_lines) > 0:
            split_lines = split_lines[len(split_lines) - num_tps :]

            # Creates subdir for each rep
            vcf_dir = sc.make_vcf_dir(input_vcf)
            sc.write_vcfs(split_lines, vcf_dir)

            # Now index and merge
            [sc.index_vcf(vcf) for vcf in glob(f"{vcf_dir}/*.vcf")]
            sc.merge_vcfs(vcf_dir)

            sc.cleanup_intermed(vcf_dir)

    except Exception as e:
        print(f"[ERROR] Couldn't process {e}")
        pass


def clean_args(ua):
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


def main(ua):
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
    ) = clean_args(ua)

    work_dir = work_dir
    vcf_dir = f"{work_dir}/vcfs"
    dumpfile_dir = f"{work_dir}/dumpfiles"
    script_dir = f"{work_dir}/scripts"

    sweeps = ["neut", "sdn", "ssv"]

    for i in [vcf_dir, dumpfile_dir, script_dir]:
        for sweep in sweeps:
            os.makedirs(f"{i}/{sweep}", exist_ok=True)

    # Inject info into SLiM script and then simulate, store params for reproducibility
    sim_params = []
    script_list = []

    if rep_range:  # Take priority
        replist = range(int(rep_range[0]), int(rep_range[1]))
    else:
        replist = range(reps)

    for rep in replist:
        for sweep in sweeps:

            # Info scraping and calculations
            raw_lines = get_slim_code(slim_file)
            raw_lines = sanitize_slim(raw_lines)

            Q, gen_time, max_years_b0, burn_in_gens, physLen = get_slim_info(raw_lines)

            # Pull from variable time of selection before sampling to make more robust
            rand_sel_gen = randomize_selTime(sel_gen, 200 / Q)

            burn_in_gens = int(round(burn_in_gens / Q))

            # Convert from earliest year from bp to gens
            end_gen = int(round((max_years_b0) / gen_time / Q))

            # Written out each step for clarity, sdn to keep track of otherwise
            # Find the earliest time before present, convert to useable times
            furthest_from_pres = max(years_sampled)
            abs_year_beg = max_years_b0 - furthest_from_pres

            sel_gen_time = (
                int(((abs_year_beg / gen_time) - rand_sel_gen) / Q)
            ) + burn_in_gens

            if sel_coeff_bounds[0] == sel_coeff_bounds[1]:
                sel_coeff = sel_coeff_bounds[0] * Q
            else:
                sel_coeff = randomize_selCoeff_uni(*sel_coeff_bounds) * Q

            recombRate = randomize_recombRate()

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
                logger.info(f"Selection coeff: {sel_coeff}")
                logger.info(f"Recomb Rate: {recombRate}")
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
                recombRate,
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

    print(f"Reps simulated: {replist}")

    for script in script_list:
        run_slim(script, slim_path)

    # Process VCFs and Cleanup
    for rep in replist:
        for sweep in sweeps:
            vcf_file = f"{vcf_dir}/{sweep}/{rep}.multivcf"
            process_vcfs(vcf_file, len(sample_sizes))
            os.remove(vcf_file)
            dumpFile = f"{dumpfile_dir}/{sweep}/{rep}.dump"
            os.remove(dumpFile)

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
