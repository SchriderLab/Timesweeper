FROM continuumio/miniconda3
LABEL org.opencontainers.image.authors="Logan Whitehouse; lswhiteh@unc.edu, Dan Schrider; drs@unc.edu"

COPY . $HOME/Timesweeper
RUN conda install -c conda-forge mamba -y \
    && mamba env create -f Timesweeper/blinx.yaml

WORKDIR $HOME/Timesweeper
ENTRYPOINT ["conda", "run", "-n", "blinx", "timesweeper"]