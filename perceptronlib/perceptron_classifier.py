import numpy as np

class PerceptronClassifier:
    '''Preceptron Binary Classifier uses Perceptron Learning Algorithm
    to do classification with two classes.

    Parameters
    ----------
    number_of_attributes : int
        The number of attributes of data set.

    Attributes
    ----------
    weights : list of float
        The list of weights corresponding with input attributes.

    errors_trend : list of int
        The number of misclassification for each training sample.

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
                    [5.7, 2.8, 4.5, 1.3]]
    Binary classes with class -1 or 1.
    >>> labels = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]
    >>> perceptronClassifier = PerceptronClassifier(4) # Four attributes
    >>> perceptronClassifier.train(samples, labels) # Training
    >>> new_data = [[6.3, 3.3, 4.7, 1.6], [4.6, 3.4, 1.4, 0.3]]
    Predict the class for the new_data
    >>> perceptronClassifier.classify(tests)
    [1, -1]
    '''
    def __init__(self, number_of_attributes: int):
        # Initialize the weigths to zero
        # The size is the number of attributes plus the bias, i.e. x_0 * w_0
        self.weights = np.zeros(number_of_attributes + 1)

        # Record of the number of misclassify for each train sample
        self.misclassify_record = []

        self._label_map = {}
        self._reversed_label_map = {}

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
            Class labels. The labels can be anything as long as it has only two types of labels.
        max_iterator : int
            The max iterator to stop the training process
            in case the training data is not converaged.
        '''
        # FIXME: add throw

        # Build the label map to map the original labels to numerical labels
        # For example, ['a', 'b', 'c'] -> {0 : 'a', 1 : 'b', 2 : 'c'}
        self._label_map = {1 : list(set(labels))[0], -1 : list(set(labels))[1]}
        self._reversed_label_map = {value : key for key, value in self._label_map.items()}

        # Transfer the labels to numerical labels
        transfered_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0
            for sample, target in zip(samples, transfered_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination >= 0.0, 1, -1)

                # use numpy.multiply to multiply element-wise
                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                # record the number of misclassification
                misclassifies += int(update != 0.0)

            if misclassifies == 0:
                break
            self.misclassify_record.append(misclassifies)

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
