import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', 
                        metavar='TRAIN_FILT_PREDICT',
                        help='Use one of the available modes, training a new\
                            model, filtering from other callsets using\
                            pre-generated npy files, or predicting from a BAM.',
                        required=True, 
                        dest='run_mode', 
                        type=str)

    args = parser.parse_args()

    return args