import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def run_xgboost():
    from experiments.xgboost import run
    print("\n" + "="*50)
    print("RUNNING XGBOOST")
    print("="*50)
    run()


def run_dnn():
    from experiments.dnn import run
    print("\n" + "="*50)
    print("RUNNING DNN")
    print("="*50)
    run()


def run_cnn():
    from experiments.cnn import run
    print("\n" + "="*50)
    print("RUNNING CNN + ATTENTION")
    print("="*50)
    run()


def main():
    parser = argparse.ArgumentParser(
        description='DHW Prediction — Train and Evaluate Models'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'dnn', 'cnn', 'all'],
        default='all',
        help='Which model to run (default: all)'
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.model == 'xgboost':
        run_xgboost()
    elif args.model == 'dnn':
        run_dnn()
    elif args.model == 'cnn':
        run_cnn()
    elif args.model == 'all':
        run_xgboost()
        run_dnn()
        run_cnn()


if __name__ == '__main__':
    main()