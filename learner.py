#!python3

from unnamed.classification import Classifier
import argparse
import sys
import time

def split_parameters(args):
    parameter_args = list()
    general_args = list()

    for arg in args:
        if arg.startswith('-p') and len(arg) > 2:
            parameter_args.append(arg)
        else:
            general_args.append(arg)

    return general_args, parameter_args

def init_parameters(parameter_args):
    parameters = dict()

    for arg in parameter_args:
        parameter_pair = arg[2:]
        parameter_pair = parameter_pair.lower()
        name, value = parameter_pair.split('=')

        if not value.isnumeric():
            value = str(value)

        else:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)

        parameters[name] = value


    return parameters


def train():
    general_args, parameter_args = split_parameters(sys.argv[2:])
    parameters = init_parameters(parameter_args)

    parser = argparse.ArgumentParser(description='train a new model and save trained model', usage='''
    learner train -i INPUT -a ALGORITHM [-o OUTPUT]  [<-pArgs>]
        
    Train and evaluate commands require algorithm name and their parameters.
    Algorithm name and their parameter follow scikit-learn document.
    In this program, these parameters specified by -p prefix.
    
    examples)
        learner train -i ./train.csv -a random_forest -pMax_depth=10 -pN_estimators=100
    ''')
    parser.add_argument('-i','--input', required=True, type=str, help='feature data (csv or spa) file path')
    parser.add_argument('-o', '--output', required=False, default=None, type=str, help='model pickle file path to store')
    parser.add_argument('-a', '--algorithm', required=True, type=str, help='classification algorithm')

    args = vars(parser.parse_args(general_args))

    data_path = args['input']
    model_path = args['output']
    algorithm = args['algorithm'].lower()

    classifier = Classifier(algorithm, parameters)
    classifier.train(data_path, model_path)

def test():
    general_args = sys.argv[2:]

    parser = argparse.ArgumentParser(description='predict test feature using trained model', usage='''
    learner test -f FILE -m MODEL

    examples)
        learner test -f ./test.csv -m ./10205132.pickle
    ''')

    parser.add_argument('-f', '--file', required=True, type=str, help='feature data (csv or spa) file path')
    parser.add_argument('-m', '--model', required=True, type=str, help='model pickle file path to load')

    args = vars(parser.parse_args(general_args))

    file_path = args['file']
    model_path = args['model']

    classifier = Classifier()
    classifier.test(model_path, file_path)


def evaluate():
    general_args, parameter_args = split_parameters(sys.argv[2:])
    parameters = init_parameters(parameter_args)

    parser = argparse.ArgumentParser(description='train a model and predict data by cross-validation', usage='''
    learner evaluate -i INPUT -a ALGORITHM [-k crossvalidation] [-l LABEL] [-s SCALE] [<-pArgs>] 

    Train and evaluate commands require algorithm name and their parameters.
    Algorithm name and their parameter follow scikit-learn document.
    In this program, these parameters specified by -p prefix.

    examples)
        learner evaluate -i ./train.csv -a random_forest -pMax_depth=10 -pN_estimators=100 -k 5
    ''')

    parser.add_argument('-i', '--input', required=True, type=str, help='feature data (csv or spa) file path')
    parser.add_argument('-a', '--algorithm', required=True, type=str, help='classification algorithm')
    parser.add_argument('-k', '--kfolds', required=False, default=3, type=int, help='k-folds')
    parser.add_argument('-n', '--nsamples', required=False, default=0, type=int,
                        help='n_sample for random sampling')
    parser.add_argument('-l', '--label', required=False, default=0, type=int, help='label index offset')
    parser.add_argument('-p', '--preprocessing', required=False, default=None, type=str, help='data preprocessing method(scale | minmax)')
    parser.add_argument('-r', '--removezero', required=False, default=False, type=bool,
                        help='remove zero vector or not')

    args = vars(parser.parse_args(general_args))

    data_path = args['input']
    algorithm = args['algorithm'].lower()
    n_folds = args['kfolds']
    label_pos = args['label']
    preprocessing_method = args['preprocessing']
    remove_zero_vector = args['removezero']
    n_samples = args['nsamples']

    classifier = Classifier(algorithm, parameters, n_samples=n_samples, label_pos=label_pos, preprocess_method=preprocessing_method, remove_zero_vector=remove_zero_vector)
    classifier.evaluate(data_path, n_folds)

def main():
    command_table = dict()
    command_table['train'] = train
    command_table['test'] = test
    command_table['evaluate'] = evaluate

    parser = argparse.ArgumentParser(description='', usage='''
    learner <command> [<args>]
    
    The commands are:
        train       train a new model and save trained model
        predict     predict feature using trained model
        evaluate    train a model and predict data by cross-validation
    ''')
    parser.add_argument('command', help='Subcommand to run')
    args = vars(parser.parse_args(sys.argv[1:2]))

    command = args.get('command', '')

    if command not in command_table:
        print('Unknown command %s'%command)
        parser.print_help()
        return

    command_table[command]()


if __name__ == '__main__':
    main()
