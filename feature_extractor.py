#!python3

import argparse
from unnamed.feature import FeatureExtractor

def process(args):
    input_path = args['input']
    output_path = args['output']
    feature = args['feature'].lower()
    n_jobs = args['jobs']
    batch_size = args['batchsize']

    fe = FeatureExtractor(input_path, output_path, feature, batch_size=batch_size, n_jobs=n_jobs)
    fe.process()


def main():
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('-i', '--input', required=True, type=str, help='input directory path')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file path')
    parser.add_argument('-f', '--feature', required=True, type=str, help='feature to extract')
    parser.add_argument('-j', '--jobs', required=False, default=1, type=int, help='No. of core')
    parser.add_argument('-b', '--batchsize', required=False, default=256, type=int, help='batch size')

    args = vars(parser.parse_args())

    process(args)


if __name__ == '__main__':
    main()
