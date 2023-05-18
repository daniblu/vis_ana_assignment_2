import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.datasets import cifar10

def main():
    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert images to grey scale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scale grey scale values
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0 

    # reshape data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    # save preprocessed data
    data = [X_train_dataset, X_test_dataset, y_train, y_test]    

    datapath = os.path.join("..", "data", "preprocessed_data.pkl")

    with open(datapath, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()