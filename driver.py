from preprocess import *
from train import *
from evaluate import *

def main():
    """
    take the file and test_size from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='data/home_data.csv', 
                        help='pass the file path, and the default path is data/home_data.csv')

    parser.add_argument("--output_path", type=str, default='result', 
                        help='save the output path, and the default path is current path')

    parser.add_argument("--test_size", type=float, default=0.33, 
                        help='take the test_size, and the default value is 0.33')

    parser.add_argument("--n_estimators", type=int, default=90, 
                        help='take the n_estimators, and the default value is 90')

    parser.add_argument("--max_features", type=int, default=80, 
                        help='take the max_features, and the default value is 80')

    parser.add_argument("--max_depth", type=int, default=20, 
                        help='take the max_depth, and the default value is 20')

    args = parser.parse_args()
    file = args.file
    output_path = args.output_path
    test_size = args.test_size
    n_estimators = args.n_estimators,
    max_features = args.max_features,
    max_depth = args.max_depth

    x_train, x_val, y_train, y_val = preprocessing(file, test_size)
    
    rf = train(x_train, y_train, n_estimators[0], max_features[0], max_depth)

    save_model(rf, output_path)
    
    y_hat, rmse, mae = evaluate(rf, x_val, y_val)

    plot(y_hat, y_val, output_path)
    

if __name__ == "__main__":
    main()
