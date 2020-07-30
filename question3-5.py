#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from eval import Evaluator
from classification import DecisionTreeClassifier
from dataset import Dataset

def combine_cross_validation(k, filename):
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
    dataset_from_file = Dataset.load_from_file(filename)
    
    file_path = "./data/" + filename
    dataset = np.loadtxt(file_path, dtype=str, delimiter=',')
    np.random.shuffle(dataset)
    subsets = np.array_split(dataset,k)

    accuracies = []

    test = Dataset.load_from_file("test.txt")
    all_predictions = np.zeros((test.labels.shape[0],k), dtype=np.object)        
    won_vote = np.zeros((test.labels.shape[0]), dtype=np.object)
    
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
        test = Dataset.load_from_file("test.txt")
        prediction = tree.predict(test.attributes)

        #Put all the predictions into a numpy array, to vote on most freq label
        for index in range(len(prediction)):
            all_predictions[index][i] = prediction[index]

        #Calculate the accuracy of each model and put into a list    
        evaluator = Evaluator()        
        confusion = evaluator.confusion_matrix(prediction, test.labels)
        a = evaluator.accuracy(confusion)
        accuracies.append(a) 
        print(accuracies)
    
    global_error_estimate = np.mean(accuracies)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

    #Create predictions with most frequent label from all k models
    for index, prediction in enumerate(all_predictions):
        
        #Ensure there are only labels in the array
        prediction = np.delete(prediction, np.argwhere(prediction == 0))

        #Get the label with the highest frequency
        unique, position = np.unique(prediction, return_inverse=True)
        count = np.bincount(position)
        pos_with_max_count = count.argmax()
        winning_label = unique[pos_with_max_count]
        won_vote[index] = winning_label

    #Calculate the accucacy of the combined model    
    print(f'WINNERS: {won_vote}')
    
    evaluator_w = Evaluator()     
    confusion_w = evaluator_w.confusion_matrix(won_vote, test.labels)
    a_w = evaluator_w.accuracy(confusion_w)
    
    return a_w

if __name__ == "__main__":
    accuracy = combine_cross_validation(10,"train_full.txt")
    print(accuracy)
