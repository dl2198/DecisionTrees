#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from eval import Evaluator
from classification import DecisionTreeClassifier

def cross_validation(k, filename):
    """
    Performs cross validation on a dataset

    Parameters
    ----------
    k : int
        number of times dataset is split
    filename : string
        name of the file to load the dataset
    
    Returns
    -------
    list of ints
        containing the accuracies of each split
    int
        global error estimate
    """
    file_path = "./data/" + filename
    dataset = np.loadtxt(file_path, dtype=str, delimiter=',')
    np.random.shuffle(dataset)
    subsets = np.array_split(dataset,k)

    accuracies = []
    
    for i in range(k):
        train = np.delete(subsets, i, axis=0)
        train = np.concatenate(train)
        train_att = train[:, :-1].astype(int)
        train_labels = train[:, -1]

        test = subsets[i]
        test_att = test[:, :-1].astype(int)
        test_labels = test[:, -1]

        tree = DecisionTreeClassifier()
        tree = tree.train(train_att, train_labels)
        prediction = tree.predict(test_att)

        evaluator = Evaluator()        
        confusion = evaluator.confusion_matrix(prediction, test_labels)
        a = evaluator.accuracy(confusion)
        accuracies.append(a) 
    
    global_error_estimate = np.mean(accuracies)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    
    return accuracies, global_error_estimate

# Example usage
if __name__ == "__main__":

    # Compute the accuracy and global error estimate on the full dataset
    accuracies, global_error = cross_validation(10,"train_full.txt")
    print(f'Accuracies: {accuracies}') 
    print(f'Global_error_estimate: {global_error}')