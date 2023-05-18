import os
import pickle
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from joblib import dump

def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="int ID to identify model")
    parser.add_argument("--hidden_layers", nargs='+', type=int, required=True, help="hidden layers design, e.g., 20 10 for two hidden layers with respective number of units")
    parser.add_argument("--max_iter", type=int, default=20, help="max number of iterations, default=20")
    parser.add_argument("--v", action="store_true", help="verbose fitting process")
    args = parser.parse_args()

    return(args)

def main(ID, hidden_layers, verbose, max_iter):
    # load data
    datapath = os.path.join("..", "data", "preprocessed_data.pkl")
    with open(datapath, 'rb') as file:
        X_train_dataset, X_test_dataset, y_train, y_test = pickle.load(file)

    # create tuple from hidden_layers argument
    hidden_layers_tuple = tuple(hidden_layers)
    
    # define classifier and fit
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers_tuple,
                    early_stopping=True,
                    verbose=verbose,
                    max_iter=max_iter).fit(X_train_dataset, y_train)
    
    # make predictions
    y_pred = clf.predict(X_test_dataset)

    # classification report
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    
    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels)
    
    # save model and txt indicating model parameters
    dump(clf, os.path.join("..", "models", f"nn{ID}.joblib"))

    txtpath = os.path.join("..", "models", f"nn{ID}.txt")
    with open(txtpath, "w") as file:
        L = [f"Hidden layers: {hidden_layers_tuple} \n", 
            f"Max iterations: {max_iter}"]
        file.writelines(L)
    
    # save report to txt file
    txtpath = os.path.join("..", "out", f"nn{ID}_report.txt")
    with open(txtpath, "w") as file:
        file.write(report)
    
if __name__ == "__main__":
    args = input_parse()
    main(ID=args.id, hidden_layers=args.hidden_layers, verbose=args.v, max_iter=args.max_iter)