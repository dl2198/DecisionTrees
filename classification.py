#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__status__ = "Prototype"

##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################


import numpy as np

from dataset import Dataset
from node import Node, Rule

attribute_names = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar",
                   "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Main methods
    ------------
    train(attributes, labels)
        Constructs a decision tree from data attributes and labels
    predict(attributes)
        Predicts the class label of sample attributes
    print()
        Prints the whole tree

    Helper methods
    --------------
    create_tree(attributes, labels)
        Recursively builds the structure of the tree
    find_best_node(attributes, labels)
        Finds the best node to the split the datase
    compute_entropy(labels)
        Computes the entropy of a list of labels
    find_most_freq_label(labels)
        Finds the most frequent label in a list of labels
    split_data(attributes, labels, split_value, split_attribute)
        Splits the dataset in two by the value of an attribute
    sort_by_attribute(attributes, labels, sorting_attribute)
        Sorts a dataset in ascending order by an attribute
    find_leaf(attributes, instance, node)
        Recursively finds the leaf containing the predicted label for an instance
    print_subtree(node,depth=0)
        Recursively prints a subtree
    """

    def __init__(self):
        self.is_trained = False

    def train(self, attributes, labels):
        """ 
        Constructs a decision tree classifier from data

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array of attributes (N is the number of instances, 
            K is the number of attributes)
        labels : numpy.array
            An N-dimensional numpy array of labels

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that attributes and labels have the same number of instances
        assert attributes.shape[0] == len(labels), \
            "Training failed. Attributes and labels must have the same number of instances."
        self.root = self.create_tree(attributes, labels)
        self.is_trained = True

        return self

    def create_tree(self, attributes, labels):
        """
        Recursively builds the structure of the tree
        
        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array of attributes (N is the number of instances, 
            K is the number of attributes)
        labels : numpy.array
            An N-dimensional numpy array of labels

        Returns
        -------
        Node
            The root of the tree
        """
        gain, parent = self.find_best_node(attributes, labels)
        most_freq_label = self.find_most_freq_label(labels)
        
        # Base cases: if met, return a leaf with the most frequent label
        # -one_label: all samples have the same label 
        # -no_more_splits: either one sample left or no information gain if split
        one_label = len(np.unique(labels)) == 1
        one_sample = np.unique(attributes).size == 1
        no_more_splits = one_sample or gain == 0

        if one_label or no_more_splits:
            leaf = Node(label=most_freq_label)
            return leaf

        # Recursion:
        # -store the majority label of the best node for pruning
        # -split the dataset
        # -call the function recursively on left and right subtrees 
        parent.majority_label = most_freq_label

        left_child, right_child = self.split_data(
            attributes, labels, parent.rule)

        parent.left = self.create_tree(
            left_child.attributes, left_child.labels)
        parent.right = self.create_tree(
            right_child.attributes, right_child.labels)

        return parent
        
    def find_best_node(self, attributes, labels):
        """
        Finds the best node to the split the dataset

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array of attributes (N is the number of instances, 
            K is the number of attributes)
        labels : numpy.array
            An N-dimensional numpy array of labels

        Returns
        -------
        int
            Information gain associated with the best node
        Node
            Best node to split the dataset
        """
        best_gain = 0
        split_value = 0
        split_attribute = 0

        attributes_count = attributes.shape[1]
        for attribute in range(attributes_count):
            attribute_values_set = set([row[attribute] for row in attributes])

            for value in attribute_values_set:
                rule = Rule(attribute, value)
                left_child, right_child = self.split_data(
                    attributes, labels, rule)

                parent_entropy = self.compute_entropy(labels)

                left_child_entropy = self.compute_entropy(left_child.labels)
                left_child_entropy *= left_child.labels.size / labels.size

                right_child_entropy = self.compute_entropy(right_child.labels)
                right_child_entropy *= right_child.labels.size / labels.size

                children_entropy = left_child_entropy + right_child_entropy

                info_gain = parent_entropy - children_entropy

                if(info_gain >= best_gain):
                    best_gain = info_gain
                    split_value = value
                    split_attribute = attribute
        
        best_rule = Rule(split_attribute, split_value)
        best_node = Node(rule=best_rule)

        return best_gain, best_node

    def compute_entropy(self, labels):
        """
        Computes the entropy of a list of labels

        Parameters
        ----------
        labels : numpy.array
            An N-dimensional numpy array of labels
        
        Returns
        -------
        int
            entropy of list of labels
        """
        entropy = 0
        labels_count = labels.size

        for label in np.unique(labels):
            label_count = np.count_nonzero(labels == label)
            probability = label_count / labels_count
            entropy += (-probability * np.log2(probability))

        return entropy

    def find_most_freq_label(self, labels):
        """
        Finds the most frequent label in a list of labels

        Parameters
        ----------
        labels : numpy.array
            An N-dimensional numpy array of labels
        
        Returns
        -------
        char
            The most frequent label
        """
        freq = 0
        most_freq = ""

        for label in labels:
            label_freq = np.count_nonzero(labels == label)

            if label_freq > freq:
                freq = label_freq
                most_freq = label

        return most_freq

    def split_data(self, attributes, labels, rule):
        """
        Splits the dataset in two according to a rule

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array of attributes (N is the number of instances, 
            K is the number of attributes)
        labels : numpy.array
            An N-dimensional numpy array of labels
        rule : Rule
            the rule by which the dataset is split

        Returns
        -------
        Dataset
            Left child : attribute smaller than value
        Dataset
            Right child : attribute greater than or equal to value
        """
        sorted_attributes, sorted_lab = self.sort_by_attribute(
            attributes, labels, rule.attribute)

        # Calculate splitting binary map
        less_than = sorted_attributes[:, rule.attribute] < rule.value

        left_child_attributes = sorted_attributes[less_than]
        left_child_labels = sorted_lab[less_than]
        left_child = Dataset(left_child_attributes, left_child_labels)

        right_child_attributes = sorted_attributes[~less_than]
        right_child_labels = sorted_lab[~less_than]
        right_child = Dataset(right_child_attributes, right_child_labels)

        return left_child, right_child

    def sort_by_attribute(self, attributes, labels, sorting_attribute):
        """
        Sorts a dataset in ascending order by an attribute

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array of attributes (N is the number of instances, 
            K is the number of attributes)
        labels : numpy.array
            An N-dimensional numpy array of labels
        sorting_attribute : int
            The attribute by which the dataset is sorted
        
        Returns
        -------
        numpy.array
            sorted_attributes : the sorted attributes
        numpy.array
            sorted_labels : the sorted labels
        """
        sorting_order = attributes[:, sorting_attribute].argsort()
        sorted_attributes = attributes[sorting_order]
        sorted_labels = labels[sorting_order]

        return sorted_attributes, sorted_labels

    def predict(self, attributes):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in attributes
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception(
                "Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        predictions = np.zeros((attributes.shape[0],), dtype=np.object)

        for instance in range(len(attributes)):
            leaf = self.find_leaf(attributes, instance, self.root)
            predictions[instance] = leaf.label

        return predictions

    def find_leaf(self, attributes, instance, node):
        """
        Recursively finds the leaf containing the predicted label for an instance

        Parameters
        ----------
        attributes : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        instance : int
            Index of the instance in the list of attributes
        node : Node
            The node of the tree being checked for a label
        
        Returns
        -------
        Node
            The leaf containing the predicted label for the instance
        """

        # Base case: if the node is a leaf, return the leaf
        if (node.is_leaf()):
            return node

        # Recurrence: 
        # -check the value of the attribute called by the node
        # -if smaller than the node value, move to the left subtree
        # -if greater than or equal to the node value, move to the right
        if (attributes[instance][node.rule.attribute] < node.rule.value):
            return self.find_leaf(attributes, instance, node.left)
        else:
            return self.find_leaf(attributes, instance, node.right)

    def print(self):
        """
        Prints the whole tree
        """
        self.print_subtree(self.root)

    def print_subtree(self, node, depth=0):
        """
        Recursively prints a subtree

        Parameters
        ----------
        node : Node
            The node being printed
        depth : int
            Counter to keep track of the depth of the node
        """
        # If there are no more branches, return
        if node is None:
            return

        # Print leaf if at terminal branch
        if node.is_leaf():
            for _ in range(depth):
                print("|", end=" ")
            print(f'Leaf: {node.label}')

        # Otherwise, print out the split point information
        elif node.is_node():
            for _ in range(depth):
                print("|", end=" ")
            print(
                f'+-----{attribute_names[node.rule.attribute]} < {node.rule.value}')

        self.print_subtree(node.left, depth+1)
        self.print_subtree(node.right, depth+1)

# Example usage
if __name__ == "__main__":
    # Create a new tree
    tree = DecisionTreeClassifier()

    # Train a tree
    train_dataset = Dataset.load_from_file("train_full.txt")
    tree = tree.train(train_dataset.attributes, train_dataset.labels)

    # Print a tree
    tree.print()

    # Predict labels 
    test_dataset = Dataset.load_from_file("test.txt")
    predictions = tree.predict(test_dataset.attributes)
    print(f'Predictions: {predictions}')
    print(f'Ground truths: {test_dataset.labels}')
