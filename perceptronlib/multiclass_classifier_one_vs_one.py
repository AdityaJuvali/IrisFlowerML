from .perceptron_classifier import PerceptronClassifier
from collections import Counter
import numpy as np

class MulticlassClassifier:
    '''Preceptron Multiclass Classifier uses One-vs-One strategy
    to do classification with multiple classes.

    Parameters
    ----------
    number_of_attributes : int
        The number of attributes of data set.

    number_of_classes : int
        The number of classes.

    Attributes
    ----------
    classifiers : list of binary Perceptron classifiers
        Since using One-vs-One strategy, requires n (n -1) / 2 binary Perceptron classifiers, 
        where n is the number of classes.

    Examples
    --------
    Two dimensions list and each sample has four attributes
     >>> samples = [[5.1, 3.5, 1.4, 0.2],
                    [4.9, 3.0, 1.4, 0.2],
                    [4.7, 3.2, 1.3, 0.2],
                    [4.6, 3.1, 1.5, 0.2],
                    [5.0, 3.6, 1.4, 0.2],
                    [5.4, 3.9, 1.7, 0.4],
                    [7.0, 3.2, 4.7, 1.4],
                    [6.4, 3.2, 4.5, 1.5],
                    [6.9, 3.1, 4.9, 1.5],
                    [5.5, 2.3, 4.0, 1.3],
                    [6.5, 2.8, 4.6, 1.5],
                    [5.7, 2.8, 4.5, 1.3],
                    [6.3, 3.3, 6.0, 2.5],
                    [5.8, 2.7, 5.1, 1.9],
                    [7.1, 3.0, 5.9, 2.1],
                    [6.3, 2.9, 5.6, 1.8],
                    [6.5, 3.0, 5.8, 2.2],
                    [7.6, 3.0, 6.6, 2.1]]
    Three classes with class 1, 2, 3.
    >>> labels = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    >>> multiclassClassifier = MulticlassClassifier(4, 3) # Four attributes and three classes
    >>> multiclassClassifier.train(SAMPLES, LABELS) # Training
    >>> new_data = [[6.3, 3.3, 4.7, 1.6], [4.6, 3.4, 1.4, 0.3], [4.9, 2.5, 4.5, 1.7]]
    Predict the class for the new_data
    >>> multiclassClassifier.classify(tests)
    [1, 2, 2]
    '''

    def __init__(self, number_of_attributes: int, number_of_classes: int):
        self.number_of_attributes = number_of_attributes
        self.number_of_classes = number_of_classes
        self.classifiers = {}

    def train(self, samples, labels, max_iterator=10):
        '''Train the model

        Parameters
        ----------
        samples : two dimensions list
            Training data set
        labels : list of labels
            Class labels. The labels can be anything.
        max_iterator : int
            The max iterator to stop the training process
            in case the training data is not converaged.
        '''
        classes = list(set(labels))
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                self.classifiers[(classes[i], classes[j])] = PerceptronClassifier(self.number_of_attributes)

        for key in self.classifiers:

            temp_labels = []
            temp_samples = []
            for label, sample in list(zip(labels, samples)):

                if label in key:
                    temp_labels.append(label)
                    temp_samples.append(sample)

            self.classifiers[key].train(temp_samples, temp_labels)

    def _get_majority(self, votes: []):
        '''Calculate the class by the majority votes.

        Parameters
        ----------
        votes : list of votes

        Return
        ------
        class that has majority votes.
        '''
        # Use numpy.unique to compute the number of each unique class
        # and convert to a tuple list
        # ['a', 'a', 'b', 'a'] -> [('a', 3), ('b', 1)]
        unique, counts = np.unique(votes, return_counts=True)
        temp_result = (zip(unique, counts))
        # Sort by values and return the class that has majority votes
        return sorted(temp_result, reverse=True, key=lambda tup : tup[1])[0][0]

    def classify(self, new_data):
        '''Classify the sample based on the trained binary perceptron models

        Parameters
        ----------
        new_data : two dimensions list
            New data to be classified

        Return
        ------
        List of int
            The list of predicted class labels.
        '''
        predict_results = []

        # Test data, i.e., new data set
        for item in new_data:
            item_result = []
            for key in self.classifiers:
                item_result += self.classifiers[key].classify([item])

            predict_results.append(self._get_majority(item_result))

        return predict_results
