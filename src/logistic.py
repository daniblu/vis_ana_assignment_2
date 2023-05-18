import os
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="int ID to identify model")
    parser.add_argument("--penalty", choices=['none', 'elasticnet', 'l1', 'l2'], help="add penalty to regression, default='l2'")
    parser.add_argument("--tol", type=float, default=0.01, help="tolerance for stopping criteria of fitting, default=0.01")
    parser.add_argument("--v", action="store_true", help="verbose fitting process")
    args = parser.parse_args()

    return(args)

def main(penalty, tol, verbose, ID):
    
    # load data
    datapath = os.path.join("..", "data", "preprocessed_data.pkl")
    with open(datapath, 'rb') as file:
        X_train_dataset, X_test_dataset, y_train, y_test = pickle.load(file)

    # define classifier and fit
    clf = LogisticRegression(penalty=penalty,
                            tol=tol,
                            verbose=verbose,
                            solver="saga",
                            multi_class="multinomial").fit(X_train_dataset, y_train)
    
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
    dump(clf, os.path.join("..", "models", f"log{ID}.joblib"))

    txtpath = os.path.join("..", "models", f"log{ID}.txt")
    with open(txtpath, "w") as file:
        L = [f"Penalty: {penalty} \n", 
            f"Tolerance: {tol}"]
        file.writelines(L)
    
    # save report to txt file
    txtpath = os.path.join("..", "out", f"log{ID}_report.txt")
    with open(txtpath, "w") as file:
        file.write(report)

if __name__ == "__main__":
    args = input_parse()
    main(penalty=args.penalty, tol=args.tol, verbose=args.v, ID=args.id)