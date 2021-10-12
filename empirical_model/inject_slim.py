from ast import dump
import os, sys, re
from numpy.lib.histograms import histogram
import pandas as pd
import numpy as np
import argparse as ap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_slim_code(slim_file):
    with open(slim_file, "r") as infile:
        raw_lines = [line.replace("\n", "") for line in infile.readlines()]

    return raw_lines


def get_years():
    """This is bespoke for the mongolian paper, need to make it more flexible."""
    raw_data = pd.read_csv(
        "Online Table 1 Newly reported ancient individuals.csv", header=0
    )
    bp_dates = raw_data.iloc[:, [5, 6, 10, 13, 18]]
    bp_dates.columns = [
        "mean_dates",
        "stddev_dates",
        "study_pass",
        "location",
        "avg_cov",
    ]

    # Filtering to only get passing samples or those that failed from contamination
    bp_dates = bp_dates[
        (bp_dates["location"] == "Mongolia")
        & (bp_dates["avg_cov"] > 1)
        & (
            (bp_dates["study_pass"] == "Yes")
            | (bp_dates["study_pass"] == "Yes and plotted in Figure 3")
            | (bp_dates["study_pass"].str.contains("contamination"))
        )
    ]
    # print(bp_dates)
    # plt.hist(pd.to_numeric(bp_dates["avg_cov"]), 20)
    # plt.title("Average Coverage for Passing Samples")
    # plt.savefig("coverage.png")

    sampled_dates = []
    for i in bp_dates.itertuples():
        # Correct for BP standardization
        sampled_dates.append(abs(1950 - int(np.random.normal(i[1], i[2]))))

    return sorted(sampled_dates)


def bin_times(gens, max_time, bin_window=5):
    """
    Creates sampling bins where close sampling points are pooled to increase sample size.
    
    Returns:

        counts - only values where there are at least one sample present
        
        bin_edges (left inclusive) - time to sample for each count
    """
    counts, edges = np.histogram(gens, range(0, max_time, bin_window))
    trimmed_edges = np.delete(edges, 0)

    good_counts = counts[counts > 0]
    good_edges = trimmed_edges[counts > 0]

    return good_counts, good_edges


def get_slim_info(slim_lines):
    Q = (
        [i for i in slim_lines if """defineConstant("Q",""" in i][0]
        .split(",")[1]
        .strip()[0]
    )

    gen_time = [
        int(s)
        for s in re.split(
            "\W+",
            [i for i in slim_lines if """defineConstant("generation_time",""" in i][0],
        )
        if s.isdigit()
    ][0]

    burn_time_mult = [
        int(s)
        for s in re.split(
            "\W+", [i for i in slim_lines if """defineConstant("burn_in",""" in i][0]
        )
        if s.isdigit()
    ][0]

    max_years_b0 = max(
        [
            int(s)
            for s in re.split(
                "\W+", [i for i in slim_lines if """defineConstant("_T",""" in i][0]
            )
            if s.isdigit()
        ]
    )

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

    return int(Q), int(gen_time), int(max_years_b0), int(round(burn_in_gens))


def inject_sweep_type(raw_lines, sweep, selCoeff, mut_rate):
    if sweep in ["hard", "soft"]:
        raw_lines.insert(
            raw_lines.index("    initializeMutationRate(mutation_rate);"),
            f"\tdefineConstant('sweep', '{sweep}');",
        )

        raw_lines.insert(
            raw_lines.index("    initializeMutationRate(mutation_rate);"),
            f"\tinitializeMutationType('m2', Q * 0.5, 'f', {selCoeff});",
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
                    cat("LOST at gen " + sim.generation + " - RESTARTING");
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


def get_argp():
    argp = ap.ArgumentParser(
        description="Injects time-series sampling into stdpopsim SLiM output script."
    )
    #!TODO Add flexible sampling for use cases that aren't ours
    argp.add_argument(
        "-i",
        "--slim-file",
        required=True,
        type=str,
        help="SLiM Script output by stdpopsim to add time-series sampling to.",
        dest="slim_file",
    )
    argp.add_argument(
        "-p",
        "--pop",
        required=False,
        type=str,
        default="p2",
        dest="pop",
        help="Label of population to sample from, will be defined in SLiM Script. Defaults to p2 for OoA CHB pop.",
    )
    argp.add_argument(
        "-n",
        "--num-samps",
        required=False,
        type=int,
        default=2,
        dest="num_samps",
        help="Number of individuals to randomly sample without replacement at each sampling point. Defaults to 2. !THIS IS ONLY FOR POST-TESTING PURPOSES, MONGOLIAN SAMPLES ARE DONE BY BINNING SAMPLING TIMES FROM PAPER.",
    )
    argp.add_argument(
        "-t",
        "--selection-generation",
        required=False,
        type=int,
        default=200,
        dest="sel_gen",
        help="Number of gens before first sampling to introduce selection in population. Defaults to 200.",
    )
    argp.add_argument(
        "-s",
        "--selection-coeff",
        required=False,
        type=float,
        default=0.05,
        dest="sel_coeff",
        help="Selection coefficient of mutation being introduced. Defaults to 0.05",
    )
    argp.add_argument(
        "-st",
        "--sweep-type",
        required=False,
        type=str,
        default="hard",
        choices=["hard", "soft", "neut"],
        dest="sweep",
        help="Introduces hard or soft sweep at the designated time with the -t flag. Leave blank or choose neut for neutral sim. Defaults to hard sweep.",
    )
    argp.add_argument(
        "--mut-rate",
        required=False,
        type=str,
        default="1.29e-8",
        dest="mut_rate",
        help="Mutation rate for mutations not being tracked for sweep detection. Defaults to 1.29e-8 as defined in stdpopsim for OoA model.",
    )
    argp.add_argument(
        "-o",
        "--out-dir",
        required=False,
        type=str,
        default="./results",
        dest="out_dir",
        help="Directory to write pop files to.",
    )
    argp.add_argument(
        "-d",
        "--dumpfile-id",
        required=False,
        type=int,
        default=np.random.randint(0, 1e6),
        dest="dumpfile_id",
        help="ID to use for dumpfile retrieval and output file labeling, optimally for SLURM job IDs. Defaults to random int between 0:1e10.",
    )
    agp = argp.parse_args()

    for i in ["pops", "slim_scripts", "dumpfiles"]:
        if not os.path.exists(os.path.join(agp.out_dir, i, agp.sweep)):
            os.makedirs(os.path.join(agp.out_dir, i, agp.sweep), exist_ok=True)

    pop_dir = os.path.join(agp.out_dir, "pops", agp.sweep)
    script_dir = os.path.join(agp.out_dir, "slim_scripts", agp.sweep)
    dump_dir = os.path.join(agp.out_dir, "dumpfiles", agp.sweep)

    print("Population output stored in:", pop_dir)
    print("Scripts stored in:", script_dir)
    print("Dumpfiles stored in:", dump_dir)

    return agp, pop_dir, script_dir, dump_dir


def main():
    # TODO Logging for different variables for clarity, especially ones skimmed from SLiM

    agp, pop_dir, script_dir, dump_dir = get_argp()

    # Info scraping and calculations
    raw_lines = get_slim_code(agp.slim_file)
    raw_lines = sanitize_slim(raw_lines)
    samp_years = get_years()

    Q, gen_time, max_years_b0, burn_in_gens = get_slim_info(raw_lines)
    burn_in_gens = int(round(burn_in_gens / Q))
    burn_in_years = burn_in_gens * gen_time

    end_gen = int(
        round((max_years_b0 + burn_in_years) / gen_time / Q)
    )  # Convert from earliest year from bc to gens

    sample_counts, binned_years = bin_times(
        samp_years, max(samp_years), bin_window=5 * gen_time
    )

    # Written out each step for clarity, hard to keep track of otherwise
    furthest_from_pres = max(binned_years)
    abs_year_beg = max_years_b0 - furthest_from_pres

    sel_gen = (int(round((abs_year_beg / gen_time) - agp.sel_gen) / Q)) + burn_in_gens

    # Logging
    print("Q Scaling Value:", Q)
    print("Gen Time:", gen_time)
    print()
    print("Burn in years:", burn_in_gens * gen_time)
    print("Burn in gens:", burn_in_gens)
    print()
    print("Number Years Simulated (post-burn):", max_years_b0)
    print("Number gens simulated (post-burn):", end_gen)
    print()
    print(
        "Number Years Simulated (inc. burn):", max_years_b0 + (burn_in_gens * gen_time)
    )
    print("Number gens simulated (inc. burn):", end_gen + burn_in_gens)
    print()
    print("Selection type:", agp.sweep)
    print("Selection start gen:", sel_gen)
    print("Number of timepoints after binning:", len(sample_counts))
    print()

    # Injection
    prepped_lines = inject_sweep_type(raw_lines, agp.sweep, agp.sel_coeff, agp.mut_rate)

    sampling_lines = inject_sampling(
        prepped_lines,
        agp.pop,
        sample_counts,
        binned_years,
        f"{pop_dir}/{agp.dumpfile_id}.pop",
    )

    selection_lines = make_sel_blocks(
        agp.sel_coeff,
        sel_gen,
        agp.pop,
        end_gen + burn_in_gens,
        f"{dump_dir}/{agp.dumpfile_id}.dump",
    )
    finished_lines = []
    finished_lines.extend(sampling_lines)
    finished_lines.extend(selection_lines)

    outfile_name = write_slim(
        finished_lines, agp.slim_file, agp.dumpfile_id, script_dir
    )

    print("Done!")
    print("Output written to:", outfile_name)


if __name__ == "__main__":
    main()
