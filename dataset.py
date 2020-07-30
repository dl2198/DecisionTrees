#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows the user to create a Dataset object to hold attributes
and labels.
"""

import numpy as np

class Dataset:
    """
    A dataset consisting of attributes and labels

    Attributes
    ----------
    attributes : numpy.array
        An N by K numpy array of attributes (N is the number of instances, 
        K is the number of attributes)
    labels : numpy.array
        An N-dimensional numpy array of labels
    
    Methods
    -------
    load_from_file(filename)
        Sets attributes and labels by reading from file
    """
    def __init__(self, attributes, labels):
        self.attributes = attributes
        self.labels = labels

    @classmethod
    def load_from_file(cls, filename):
        """
        Sets attributes and labels by reading from file

        Parameters
        ----------
        filename: string
            Name of the file
        """
        file_path = "./data/" + filename
        all_data = np.loadtxt(file_path, dtype=str, delimiter=',')
        labels = all_data[:, -1]
        attributes = all_data[:, :-1].astype(int)
        return cls(attributes, labels)


# Example usage
if __name__ == "__main__":

    # Example 1 : load_from_file(filename)
    print("Creating a dataset by loading from file:")
    filename = "toy.txt"
    loaded_dataset = Dataset.load_from_file(filename)
    print(f"Attributes: {loaded_dataset.attributes}")
    print(f"Labels: {loaded_dataset.labels}")
    print()

    # Example 2 : constructor
    print("Creating a dataset by calling the constructor:")
    attributes = loaded_dataset.attributes
    labels = loaded_dataset.labels
    dataset_with_constructor = Dataset(attributes,labels)
    print(f"Attributes: {dataset_with_constructor.attributes}")
    print(f"Labels: {dataset_with_constructor.labels}")
