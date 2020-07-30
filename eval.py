#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np

from dataset import Dataset
from classification import DecisionTreeClassifier

class Evaluator(object):
    """ 
    Class to perform evaluation

    Methods
    -------
    confusion_matrix(prediction,annotation,class_labels=None)
        Computes the confusion matrix
    accuracy(confusion)
        Computes the accuracy given a confusion matrix
    precision(confusion,rounding=True)
        Computes the precision score per class given a confusion matrix
        Also returns the macro-averaged precision across classes
    recall(confusion, rounding=True)
        Computes the recall score per class given a confusion matrix
        Also returns the macro-averaged recall across classes
    f1_score(confusion)
        Computes the f1 score per class given a confusion matrix
        Also returns the macro-averaged f1-score across classes
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ 
        Computes the confusion matrix

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros(
            (len(class_labels), len(class_labels)), dtype=np.int)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
        
        for i in range(len(annotation)):
            ground_truth_label = annotation[i]
            predicted_label = prediction[i]

            ground_truth_position = np.where(
                class_labels == ground_truth_label)
            predicted_position = np.where(
                class_labels == predicted_label)

            confusion[ground_truth_position[0][0]
                      ][predicted_position[0][0]] += 1

        return confusion

    def accuracy(self, confusion):
        """ 
        Computes the accuracy given a confusion matrix

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        # feel free to remove this
        accuracy = 0.0

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################
        
        correct_predictions = np.trace(confusion)
        all_predictions = np.sum(confusion)

        accuracy = correct_predictions / all_predictions
        accuracy = round(accuracy, 4)

        return accuracy

    def precision(self, confusion, rounding=True):
        """ 
        Computes the precision score per class given a confusion matrix

        Also returns the macro-averaged precision across classes

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        total_predictions = np.sum(confusion, axis=0)

        for i in range(len(confusion)):
            correct_predictions = confusion[i][i]
            p[i] = correct_predictions / total_predictions[i]
        
        macro_p = np.mean(p)
        
        if rounding:
            p = np.around(p, decimals=4)
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
            macro_p = round(macro_p, 4)
        
        return (p, macro_p)

    def recall(self, confusion, rounding=True):
        """ 
        Computes the recall score per class given a confusion matrix

        Also returns the macro-averaged recall across classes

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        total_predictions = np.sum(confusion, axis=1)

        for i in range(len(confusion)):
            correct_predictions = confusion[i][i]
            r[i] = correct_predictions / total_predictions[i]
      
        macro_r = np.mean(r)
        
        if rounding:
            r = np.around(r, decimals=4)
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
            macro_r = round(macro_r, 4)

        return (r, macro_r)

    def f1_score(self, confusion):
        """ 
        Computes the f1 score per class given a confusion matrix

        Also returns the macro-averaged f1-score across classes

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        p, _ = self.precision(confusion, rounding=False)
        r, _ = self.recall(confusion, rounding=False)

        for i in range(len(f)):
            num = p[i] * r[i]
            den = p[i] + r[i]
            f[i] = num / den
            f[i] *= 2
        
        # You will also need to change this
        macro_f = np.mean(f)

        f = np.around(f, decimals=4)
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        macro_f = round(macro_f, 4)

        return (f, macro_f)

# Example usage
if __name__ == "__main__":

    # Create an evaluator and load the datasets
    evaluator = Evaluator()
    training_dataset = Dataset.load_from_file("train_full.txt")
    test_dataset = Dataset.load_from_file("test.txt")

    # Create a tree, train it and test it on the datasets
    tree = DecisionTreeClassifier()
    trained_tree = tree.train(training_dataset.attributes, training_dataset.labels)
    prediction = tree.predict(test_dataset.attributes)

    # Compute the confusion matrix
    confusion = evaluator.confusion_matrix(prediction, test_dataset.labels)
    print(f'Confusion matrix: {confusion}')

    # Compute the accuracy
    a = evaluator.accuracy(confusion)
    print(f'Accuracy: {a}')

    # Compute the precision
    p, macro_p = evaluator.precision(confusion)
    print(f'Precision: {p}') 
    print(f'Macro precision: {macro_p}')

    # Compute the recall
    r, macro_r = evaluator.recall(confusion)
    print(f'Recall: {r}')
    print(f'Macro recall: {macro_r}')

    #Compute the f1 score
    f1, macro_f1 = evaluator.f1_score(confusion)
    print(f'F1 score: {f1}')
    print(f'Macro F1 score: {macro_f1}')