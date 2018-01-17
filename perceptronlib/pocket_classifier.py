import numpy as np
import pandas as pd

class Pocket:
    '''This class keeps the best weights in the learning process.

    Parameters
    ----------
    number_of_attributes : int
        The number of attributes of the data set.

    Attributes
    ----------
    best_weights: list of float
        The best list of weights so far.

    misclassify_count: int
        The number of misclassification corresponding to the best weights.
    '''
    def __init__(self, number_of_attributes):
        self.best_weights = np.zeros(number_of_attributes + 1)
        self.misclassify_count = -1
        #FIXME add @property

class PocketClassifier:
    '''Pocket Binary Classifier uses modified Perceptron Learning Algorithm called
        Pocket Learning Algorithm to classify two classes data.

    Parameters
    ----------
    number_of_attributes : int
        The number of attributes of the data set.

    class_labels : tuple of the class labels
        The class labels can be anything as long as it has only two types of labels.

    Attributes
    ----------
    pocket: Pocket
        The pocket contains the best weights so far and
        the number of misclassified sample according to the current best weights.

    weights : list of float
        The list of weights corresponding input attributes.

    misclassify_record : list of int
        The number of misclassification for each training sample.
    '''
    def __init__(self, number_of_attributes: int, class_labels: ()):
        # Initialize the pocket class
        self.pocket = Pocket(number_of_attributes)
        # Initialize the weights to zero
        # The size is the number of attributes plus the bias, i.e. x_0 * w_0
        self.weights = np.zeros(number_of_attributes + 1)

        # Record of the number of misclassify for each train sample
        self.misclassify_record = []

        # Build the label map to map the original labels to numerical labels
        # For example, ['a', 'b'] -> {0: 'a', 1: 'b'}
        self._label_map = {1: class_labels[0], -1: class_labels[1]}
        self._reversed_label_map = {class_labels[0]: 1, class_labels[1]: -1}

    def _linear_combination(self, sample):
        '''linear combination of sample and weights'''
        return np.inner(sample, self.weights[1:])

    def train(self, samples, labels, max_iterator=10):
        '''Train the model

        Parameters
        ----------
        samples : two dimensions list
            Training data set
        labels : list of labels
            The class labels of the training data
        max_iterator : int
            The max iterator to stop the training process.
        '''
         # Transfer the labels to numerical labels
        transferred_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0
            for sample, target in zip(samples, transferred_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination >= 0.0, 1, -1)

                # use numpy.multiply to multiply element-wise
                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                # record the number of misclassification
                misclassifies += int(update != 0.0)

            if (self.pocket.misclassify_count == -1) or \
                (self.pocket.misclassify_count > misclassifies) or (misclassifies == 0):

                self.pocket.best_weights = self.weights
                self.pocket.misclassify_count = misclassifies

            if misclassifies == 0:
                break

            self.misclassify_record.append(self.pocket.misclassify_count)

    def classify(self, new_data):
        '''Classify the sample based on the trained weights

        Parameters
        ----------
        new_data : two dimensions list
            New data to be classified

        Return
        ------
        List of int
            The list of predicted class labels.
        '''
        predicted_result = np.where((self._linear_combination(new_data) + self.weights[0]) >= 0.0, 1, -1)
        return [self._label_map[item] for item in predicted_result]

'''
if __name__ == '__main__':
    samples = [[5.1, 3.5, 1.4, 0.2],
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
                    [5.7, 2.8, 4.5, 1.3]]
    labels = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]

    pocket_classifier = PocketClassifier(4, [-1, 1 ])

    pocket_classifier.train(samples, labels)

    print(pocket_classifier.weights)
    print(pocket_classifier.pocket.best_weights)
'''