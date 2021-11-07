from glob import glob
import os
import sys
from haplotypes import samps_to_gens
from tqdm import tqdm

base_dir = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps-code/empirical_model/OoA_stdpopsim/slimulations"
schema_name = "mongolian_ooa_stdpopsim"

# Preprocessing on config file, make sure all necessary args are present in a way that makes sense
print("Base input directory:", config["base_sim_dir"])


if config["schema_name"] is not None:
    schema_name = config["schema_name"]
else:
    schema_name = "schema-" + "-".join(
        [str(i) for i in sample_gens] + "_gens_" + config["sample_size"] + "_samps"
    )

print("Schema name:", schema_name)

# Get list of lowest-level subdirs containing replicates
SIMTYPES, BATCHES, _, __ = glob_wildcards(
    f"{base_dir}/{{simtype}}/{{batch}}/{{popsf}}/{{popd}}.pop"
)

SIMTYPES = list(set(SIMTYPES))

rule all:
    input:
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps_predictions.csv",
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps1Samp_predictions.csv",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps_training.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps1Samp_training.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps_confmat.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps1Samp_confmat.png",
        f"{base_dir}/{schema_name}/images/summarized-{schema_name}-rocs.png",
        f"{base_dir}/{schema_name}/images/summarized-{schema_name}-confmat.png",
        f"{base_dir}/{schema_name}/images/summarized-1Samp-{schema_name}-confmat.png",
        f"{base_dir}/{schema_name}/images/{schema_name}.mean.all.png",
        f"{base_dir}/{schema_name}/images/{schema_name}.mean.zoomed.png"


rule createNpzFiles:
    input:
        f"{base_dir}/{{simtype}}/pops"
    output:
        f"{base_dir}/{{simtype}}/hfs_{schema_name}.npz"
    params:
        outdir=f"{base_dir}/{{simtype}}"
    shell:
        f"""
        conda activate blinx
        python {config["src_dir"]}/haplotypes.py -i {input} \
            --out-dir {params.outdir} \
            --schema-name {schema_name}
        """

rule mergeNpzs:
    input:
        expand(
            f"{base_dir}/{{simtype}}/hfs_{schema_name}.npz",
            simtype=SIMTYPES,
        )
    output:
        f"{base_dir}/{schema_name}/{schema_name}.npz"
    shell:
        f"""
        conda activate blinx
        python {config["src_dir"]}/merge_npzs.py {base_dir}/{schema_name}/{schema_name}.npz {input}
        """

rule trainModels:
    input:
        f"{base_dir}/{schema_name}/{schema_name}.npz",
    output:
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps_predictions.csv",
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps1Samp_predictions.csv",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps_training.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps1Samp_training.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps_confmat.png",
        f"{base_dir}/{schema_name}/images/{schema_name}_TimeSweeperHaps1Samp_confmat.png"
    shell:
        f"""
        conda deactivate
        conda activate blinx
        python {src_dir}/hap_networks.py train -i {input} -n {schema_name}
        """

rule combineResults:
    input:
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps_predictions.csv",
        f"{base_dir}/{schema_name}/{schema_name}_TimeSweeperHaps1Samp_predictions.csv"
    output:
        f"{base_dir}/{schema_name}/images/summarized-{schema_name}-rocs.png",
        f"{base_dir}/{schema_name}/images/summarized-{schema_name}-confmat.png",
        f"{base_dir}/{schema_name}/images/summarized-1Samp-{schema_name}-confmat.png",
    shell:
        f"""
        conda activate blinx
        python {config["src_dir"]}/summarize_3class.py {base_dir}/{schema_name}/images {schema_name} {input}
        """

rule plotHaps:
    input:
        f"{base_dir}/{schema_name}/{schema_name}.npz"
    output:
        f"{base_dir}/{schema_name}/images/{schema_name}.mean.all.png",
        f"{base_dir}/{schema_name}/images/{schema_name}.mean.zoomed.png"
    shell:
        f"""
        conda deactivate
        conda activate blinx
        python {config["src_dir"]}/plotting/plot_hap_spec.py {base_dir}/{schema_name}/{schema_name}.npz
        """