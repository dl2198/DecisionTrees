#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eval import Evaluator
from dataset import Dataset
from node import Node, Rule
from classification import DecisionTreeClassifier

def pruning(tree, dataset):
    """
    Performs pruning of a trained decision tree classifier

    Parameters
    ----------
    tree : DecisionTreeClassifier
        the tree to be pruned
    dataset : Dataset
        the dataset to be used for pruning
    """
    if not tree.is_trained:
        raise Exception(
            "Decision Tree classifier has not yet been trained.")

    prune(tree.root, dataset)
    
def prune(node, dataset):
    """
    Recursively prunes a decision tree classifier

    Parameters
    ----------
    root : Node
        the node being considered for pruning
    dataset: Dataset
        the dataset being used for pruning
    """
    if not node.is_leaf():
        two_leaves = node.right.is_leaf() and node.left.is_leaf()

        if not two_leaves:
            if (node.left.is_node()):
                prune(node.left, dataset)
            if (node.right.is_node()):
                prune(node.right, dataset)

        two_leaves = node.right.is_leaf() and node.left.is_leaf()

        if two_leaves:
            annotation = dataset.labels
            attributes = dataset.attributes

            # Try and prune current node
            # Calculate accuracy before pruning
            evaluator = Evaluator()
            predictions_before = tree.predict(attributes)
            confusion = evaluator.confusion_matrix(predictions_before, annotation)
            accuracy_before = evaluator.accuracy(confusion)

            # Store leaves and rule temporarily
            temp_label_left = node.left.label
            temp_label_right = node.right.label
            temp_rule = node.rule

            # Prune current node
            node.label = node.majority_label
            node.left.label = None
            node.right.label = None
            node.rule = None

            # Calculate accuracy after pruning
            predictions_after = tree.predict(attributes)
            confusion = evaluator.confusion_matrix(predictions_after, annotation)
            accuracy_after = evaluator.accuracy(confusion)

            # Restore node if accuracy dropped
            if (accuracy_after < accuracy_before):
                node.left.label = temp_label_left
                node.right.label = temp_label_right
                node.label = None
                node.rule = temp_rule

# Example usage
if __name__ == "__main__":

    # Create and train a tree
    training_dataset = Dataset.load_from_file("train_full.txt")
    tree = DecisionTreeClassifier()
    tree = tree.train(training_dataset.attributes, training_dataset.labels)
    
    # Print tree before pruning
    tree.print()

    # Evaluate predictions before pruning
    evaluator = Evaluator()
    validation_dataset = Dataset.load_from_file("validation.txt")
    predictions_before = tree.predict(validation_dataset.attributes)
    confusion = evaluator.confusion_matrix(
        predictions_before, validation_dataset.labels)
    accuracy_before = evaluator.accuracy(confusion)
    print(f'Accuracy before: {accuracy_before}')

    # Perform pruning
    pruning(tree, validation_dataset)

    # Print tree after pruning
    tree.print()

    # Evaluate predictions after pruning
    predictions_after = tree.predict(validation_dataset.attributes)
    confusion = evaluator.confusion_matrix(
        predictions_after, validation_dataset.labels)
    accuracy_after = evaluator.accuracy(confusion)
    print(f'Accuracy after: {accuracy_after}')
