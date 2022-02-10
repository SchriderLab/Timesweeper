import uaarse as ap
import logging
import os
import re

import numpy as np

from timesweeper import read_config


def get_slim_code(slim_file):
    with open(slim_file, "r") as infile:
        raw_lines = [line.replace("\n", "") for line in infile.readlines()]

    return raw_lines


def get_slim_info(slim_lines):
    def get_ints(lines, query):
        """Splits a line by regex split and grabs all digits. Returns as list."""
        # Uses max to handle max_years_b0, but might need to check if that causes other parsing issues
        return [
            int(s)
            for s in re.split("\W+", [i for i in lines if query in i][0])
            if s.isdigit()
        ]

    # format: off
    Q = get_ints(slim_lines, """defineConstant("Q",""")[0]
    gen_time = get_ints(slim_lines, """defineConstant("generation_time",""")[0]
    burn_time_mult = get_ints(slim_lines, """defineConstant("burn_in",""")[0]
    max_years_b0 = max(get_ints(slim_lines, """defineConstant("_T","""))
    physLen = get_ints(slim_lines, """defineConstant("chromosome_length",""")[0]
    # format: on

    pop_sizes_line = [i for i in slim_lines if "_N" in i][0]
    pop_sizes_start = slim_lines.index(pop_sizes_line) + 2
    # The 4,3 will need to be more flexible
    pop_sizes_end = (
        slim_lines[pop_sizes_start:].index(
            '    defineConstant("num_epochs", length(_T));'
        )
        - 2
        + pop_sizes_start
    )  # End of definition idx, will be end of sampling ep array

    sizes = []
    for i in slim_lines[pop_sizes_start:pop_sizes_end]:
        sizes.append([int(s) for s in re.split("\W+", i) if s.isdigit()])
    first_biggest = max([i[0] for i in sizes])
    burn_in_gens = first_biggest * burn_time_mult

    return Q, gen_time, max_years_b0, round(burn_in_gens), physLen


def inject_sweep_type(raw_lines, sweep, selCoeff, mut_rate):
    if sweep in ["hard", "soft", "neut"]:
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

        raw_lines[raw_lines.index('    defineConstant("mutation_rate", Q * 0);')] = (
            f'    defineConstant("mutation_rate", Q * {mut_rate});',
        )[
            0
        ]  # Weird BLACK formatting issue casts as a tuple

    return raw_lines


def sanitize_slim(raw_lines):
    raw_lines.pop(raw_lines.index("    sim.treeSeqOutput(trees_file);"))
    raw_lines.pop(raw_lines.index("    initializeTreeSeq();"))
    raw_lines.pop(
        raw_lines.index("""            "inds=p"+pop+".sampleIndividuals("+n+"); " +""")
    )
    return raw_lines


def inject_sampling(raw_lines, pop, samp_counts, gens, outfile_path):
    samp_eps_line = [i for i in raw_lines if "sampling_episodes" in i][0]
    samp_eps_start = raw_lines.index(
        samp_eps_line
    )  # Start of sampling_episodes constant idx
    samp_eps_end = (
        raw_lines[samp_eps_start:].index("    ), c(3, 3)));") + samp_eps_start
    )  # End of definition idx, will be end of sampling ep array

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
            ] = f"""\t\t\t"{pop}.outputSample("+n+", replace=F, filePath='{outfile_path}', append=T);}}","""

    finished_lines = raw_lines[:samp_eps_start]
    finished_lines.extend(new_lines)
    finished_lines.extend(raw_lines[samp_eps_end + 1 :])

    return finished_lines


def make_sel_blocks(sel_coeff, sel_gen, pop, end_gen, dumpFileName):
    intro_block = f"""\n{sel_gen} {{
        if (sweep == "hard")
        {{    
            // save the state of the simulation 
            cat("SAVING TO " + "{dumpFileName}" + " at generation " + sim.generation);
            sim.outputFull("{dumpFileName}");
            // introduce the sweep mutation
            target = sample({pop}.genomes, 1);
            cat("INTRODUCED MUTATION at gen " + sim.generation); //" with 2Ns = " + 2*subpopSize*{sel_coeff});
            target.addNewDrawnMutation(m2, asInteger(chromosome_length/2));
        }}
    }}

    """

    soft_block = f"""{sel_gen} {{
        if (sweep == "soft")
        {{
            muts = sim.mutationsOfType(m1);
            if (size(muts))
            {{
                mut = NULL;
                minDist = chromosome_length+1;
                for (m in muts)
                {{
                    dist = abs(m.position-asInteger(chromosome_length/2));
                    if (dist < minDist)
                    {{
                        minDist = dist;
                        mut = m;
                    }}
                }}
                cat("chosen mut:" + mut.id);
                mut.setMutationType(m2);
                mut.setSelectionCoeff(selCoeff);
                cat("Chose polymorphism at position " + mut.position + " and frequency " + sim.mutationFrequencies({pop}, mut) + " to become beneficial at generation " + sim.generation);
                // save the state of the simulation 
                cat("SAVING TO " + "{dumpFileName}" + " at generation " + sim.generation);
                sim.outputFull("{dumpFileName}");
            }}
            else
            {{
                cat("Failed to switch from neutral to beneficial at gen " + sim.generation);
            }}
        }}
    }}
    """

    check_block = f"""{sel_gen}:{end_gen} late(){{
        if (sweep == "hard" | (sweep == "soft"))
        {{
            fixed = (sum(sim.substitutions.mutationType == m2) == 1);
            if (fixed)
            {{
                cat("FIXED in pop 1 at gen " + sim.generation);
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
                        cat("RE-INTRODUCED MUTATION at gen " + sim.generation); //" with 2Ns = " + 2*subpopSize*selCoeff);
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


def write_slim(finished_lines, slim_file, dumpfile_id, out_dir):
    filename = os.path.basename(slim_file).split(".")[0]
    new_file_name = os.path.join(out_dir, f"modded.{dumpfile_id}.{filename}.slim")
    with open(new_file_name, "w") as outfile:
        for line in finished_lines:
            outfile.write(line + "\n")

    return new_file_name


def get_ua():
    agp = ap.ArgumentParser(
        description="Injects time-series sampling into stdpopsim SLiM output script."
    )
    #!TODO Add flexible sampling for use cases that aren't ours
    subparsers = agp.add_subparsers(dest="config_format")
    subparsers.required = True
    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        "-y",
        "--yaml",
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
        "-p",
        "--pop",
        required=False,
        type=str,
        default="p2",
        dest="pop",
        help="Label of population to sample from, will be defined in SLiM Script. Defaults to p2.",
    )
    cli_parser.add_argument(
        "-n",
        "--num-samps",
        required=True,
        type=int,
        nargs="+",
        dest="num_samps",
        help="Number of individuals to sample without replacement at each sampling point. Will be multiplied by 2 to sample both chroms from slim. Must match the number of entries in the -y flag.",
    )
    cli_parser.add_argument(
        "-p",
        "--ploidy",
        dest="ploidy",
        help="Ploidy of organism being sampled.",
        default="2",
        type=int,
    )
    cli_parser.add_argument(
        "-y",
        "--years-sampled",
        required=True,
        type=int,
        nargs="+",
        dest="years_sampled",
        help="Years BP (before 1950) that samples are estimated to be from. Must match the number of entries in the -n flag.",
    )
    cli_parser.add_argument(
        "-t",
        "--selection-generation",
        required=False,
        type=int,
        default=200,
        dest="sel_gen",
        help="Number of gens before first sampling to introduce selection in population. Defaults to 200.",
    )
    cli_parser.add_argument(
        "-s",
        "--selection-coeff",
        required=False,
        type=float,
        default=0.05,
        dest="sel_coeff",
        help="Selection coefficient of mutation being introduced. Defaults to 0.05",
    )
    cli_parser.add_argument(
        "--sweep-type",
        required=False,
        type=str,
        default="hard",
        choices=["hard", "soft", "neut"],
        dest="sweep",
        help="Introduces hard or soft sweep at the designated time with the -t flag. Leave blank or choose neut for neutral sim. Defaults to hard sweep.",
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
        "-o",
        "--out-dir",
        required=False,
        type=str,
        default="./results",
        dest="out_dir",
        help="Directory to write pop files to.",
    )
    agp.add_argument(
        "-d",
        "--dumpfile-id",
        required=False,
        type=int,
        default=np.random.randint(0, 1e6),
        dest="dumpfile_id",
        help="ID to use for dumpfile retrieval and output file labeling, optimally for SLURM array job IDs. Defaults to random int between 0:1e10.",
    )
    ua = agp.parse_args()

    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        (
            slim_file,
            pop,
            num_samps,
            ploidy,
            years_sampled,
            sel_gen,
            sel_coeff,
            sweep,
            mut_rate,
            out_dir,
            dumpfile_id,
        ) = (
            yaml_data["slimfile"],
            yaml_data["pop"],
            yaml_data["sample sizes"],
            yaml_data["ploidy"],
            yaml_data["years"],
            yaml_data["selection gen"],
            yaml_data["selection coeff"],
            yaml_data["sweep"],
            yaml_data["mut rate"],
            yaml_data["output dir"],
            ua.dumpfile_id,
        )
    elif ua.config_format == "cli":
        input_vcf, samp_sizes, ploidy = (
            ua.input_vcf,
            ua.samp_sizes,
            ua.ploidy,
        )

    if len(agp.num_samps) != len(agp.years_sampled):
        logging.error(
            "Number of args supplied for generations and sample sizes don't match up. Please double check your values."
        )
        raise ValueError()

    if agp.years_sampled[0] < agp.years_sampled[-1]:
        logging.warning(
            "Gens were not supplied in earliest to latest order, sorting and flipping."
        )
        years_sampled = sorted(agp.years_sampled)[::-1]
        sample_sizes = sorted(agp.num_samps)[::-1]
    else:
        years_sampled = agp.years_sampled
        sample_sizes = agp.num_samps

    for i in ["pops", "slim_scripts", "dumpfiles"]:
        if not os.path.exists(os.path.join(agp.out_dir, i)):
            os.makedirs(os.path.join(agp.out_dir, i), exist_ok=True)

    pop_dir = os.path.join(agp.out_dir, "pops")
    script_dir = os.path.join(agp.out_dir, "slim_scripts")
    dump_dir = os.path.join(agp.out_dir, "dumpfiles")

    print("Population output stored in:", pop_dir)
    print("Scripts stored in:", script_dir)
    print("Dumpfiles stored in:", dump_dir)

    # Make sure the id hasn't been used already in case it's a randomly-generated one
    dumpid = agp.dumpfile_id
    while os.path.exists(os.path.join(dump_dir, str(dumpid) + ".dump")):
        dumpid = np.random.randint(0, 1e6)

    return (
        agp,
        pop_dir,
        script_dir,
        dump_dir,
        dumpid,
        years_sampled,
        [i * agp.ploidy for i in sample_sizes],
    )


def main():
    agp, pop_dir, script_dir, dump_dir, dumpid, years_sampled, sample_sizes = get_ua()

    # Info scraping and calculations
    raw_lines = get_slim_code(agp.slim_file)
    raw_lines = sanitize_slim(raw_lines)

    Q, gen_time, max_years_b0, burn_in_gens, physLen = get_slim_info(raw_lines)
    burn_in_gens = int(round(burn_in_gens / Q))
    burn_in_years = burn_in_gens * gen_time

    end_gen = int(
        round((max_years_b0 + burn_in_years) / gen_time / Q)
    )  # Convert from earliest year from bp to gens

    # Written out each step for clarity, hard to keep track of otherwise
    # Find the earliest time before present, convert to useable times
    furthest_from_pres = max(years_sampled)
    abs_year_beg = max_years_b0 - furthest_from_pres

    sel_gen = (int(round((abs_year_beg / gen_time) - agp.sel_gen) / Q)) + burn_in_gens

    # Logging - is there a cleaner way to do this?
    print("Timesweeper SLiM Injection")
    print("Q Scaling Value:", Q)
    print("Gen Time:", gen_time)
    print("Simulated Chrom Length:", physLen)
    print()
    print("Burn in years:", burn_in_gens * gen_time)
    print("Burn in gens:", burn_in_gens)
    print()
    print("Number Years Simulated (post-burn):", max_years_b0)
    print("Number gens simulated (post-burn):", end_gen - burn_in_gens)
    print()
    print(
        "Number Years Simulated (inc. burn):", max_years_b0 + (burn_in_gens * gen_time)
    )
    print("Number gens simulated (inc. burn):", end_gen)
    print()
    print("Selection type:", agp.sweep)
    print("Selection start gen:", sel_gen)
    print("Number of timepoints:", len(years_sampled))
    print()
    print("Sample sizes:", " ".join([str(i) for i in sample_sizes]))
    print(
        "Years before present (1950) sampled:",
        " ".join([str(i) for i in years_sampled]),
    )
    print(
        "Gens before present (1950) sampled:",
        " ".join([str(int(i / gen_time / Q)) for i in years_sampled]),
    )

    # Injection
    prepped_lines = inject_sweep_type(raw_lines, agp.sweep, agp.sel_coeff, agp.mut_rate)

    sampling_lines = inject_sampling(
        prepped_lines, agp.pop, sample_sizes, years_sampled, f"{pop_dir}/{dumpid}.pop",
    )

    selection_lines = make_sel_blocks(
        agp.sel_coeff, sel_gen, agp.pop, end_gen, f"{dump_dir}/{dumpid}.dump",
    )
    finished_lines = []
    finished_lines.extend(sampling_lines)
    finished_lines.extend(selection_lines)

    outfile_name = write_slim(finished_lines, agp.slim_file, dumpid, script_dir)

    print("Done!")
    print("Output written to:", outfile_name)


if __name__ == "__main__":
    main()
