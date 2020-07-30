#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Node(object):
    """
    A node of the decision tree classifier
    If node is a leaf, all attributes are None but label
    If node is not a leaf, all attributes are set but label

    Attributes
    ----------
    rule : Rule
        the rule defining how the node splits a dataset
    left : Node
        the root node of the left subtree
    right : Node
        the root node of the right subtree
    label : char
        the label defining a leaf
    most_freq_lab : char
        the most freqent label of the dataset at the node

    Methods
    -------
    is_leaf()
        Checks if node is a leaf
    is_node()
        Checks if node is not a leaf
    """
    def __init__(self, rule=None, left=None, right=None, label=None, most_freq_lab=None):
        self.rule = rule
        self.left = left
        self.right = right
        self.label = label
        self.most_freq_lab = most_freq_lab

    def is_leaf(self):
        """
        Checks if node is a leaf

        Returns
        -------
        bool
            True : if node is a leaf (has a label and no rule)
        bool
            False : if node is not a leaf (doesn't have a label)
        """
        if self.label != None and self.rule == None:
            return True
        return False

    def is_node(self):
        """
        Checks if node is not a leaf

        Returns
        -------
        bool
            True : if node is not a leaf
        bool
            False : if node is a leaf
        """
        if self.rule != None and self.label == None:
            return True
        return False

class Rule(object):
    """
    A rule that defines how a node splits a dataset.
    
    Attributes
    ----------
    attribute : int
        the attribute being used for the split
    value : int
        the threshold value used for the split
    """
    def __init__(self, attribute,value):
        self.attribute = attribute
        self.value = value