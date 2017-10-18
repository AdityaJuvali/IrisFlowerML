from perceptron_classifier import PerceptronClassifier

class MulticlassClassifier:
    '''Preceptron Multiclass Classifier uses One-vs-All strategy
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
        Since using One-vs-All strategy, each class requires a binary Perceptron classifier.

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
        self.classifiers = [PerceptronClassifier(number_of_attributes) for _ in range(number_of_classes)]

        self._label_map = {}
        self._reversed_label_map = {}

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
        unique_labels = list(set(labels))

        # Build the label map to map the original labels to numerical labels
        # For example, ['a', 'b', 'c'] -> {0 : 'a', 1 : 'b', 2 : 'c'}
        self._label_map = dict(zip([key for key in range(len(unique_labels))], unique_labels))
        # Reversed label map, so it can be searched by value
        self._reversed_label_map = {value : key for key, value in self._label_map.items()}

        # Transfer the labels to numerical labels
        transfered_labels = [self._reversed_label_map[index] for index in labels]

        for class_index in range(self.number_of_classes):
            binary_labels = ['One' if item == class_index else 'All' for item in transfered_labels]
            self.classifiers[class_index].train(samples, binary_labels, max_iterator)

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
        predicted_results = []

        # Test data, i.e., new data set
        for item in new_data:
            for class_index in range(self.number_of_classes):
                if self.classifiers[class_index].classify([item])[0] == 'One':
                    predicted_results.append(class_index)
                    break
        return [self._label_map[item] for item in predicted_results]
