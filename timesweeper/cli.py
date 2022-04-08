import make_training_features as mtf
import argparse
import sys


def ts_main():
    agp = argparse.ArgumentParser(description="Timesweeper CLI")
    agp.add_argument("mode", choices=["train"])
    agp.add_argument("args", nargs="?", default=["-h"])
    ua = agp.parse_args()

    if ua.mode == "train":
        mua = mtf.parse_ua(ua.args)
        mtf.main(mua)


if __name__ == "__main__":
    ts_main()
