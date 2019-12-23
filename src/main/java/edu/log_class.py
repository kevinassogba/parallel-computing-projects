import numpy as np

def gradientDescent(dataset, learning_rate, max_iter):
    """
        This method computes the gradient and updates the weights
    """
    num_values = int(len(dataset[0]))
    weight = [1] * num_values

    for iter in range(max_iter):
        gradients = [0] * num_values
        for row in dataset:
            predicted = _sigmoid(row, weight)
            error = row[-1] - predicted

            for feature in range(1, len(gradients)):
                gradients[feature] = error * row[feature - 1]

            for index in range(len(weight)):
                weight[index] += learning_rate * gradients[index]

    return weight

def _sigmoid(record, weight):

    """
        Thic function computes the sigmoid function and
        return the predicted value
    """
    res = weight[0]
    weights = weight[1:]
    if len(record) == len(weight):
        X = list(record[:-1])
    else:
        X = list(record)
    for index in range(len(weights)):
        res += X[index] * weights[index]
    return 1/(1 + np.exp(-res))

def get_class(prediction, task = ''):

    if task != '':
        return find_class_train(prediction)
    else:
        collection = np.array(prediction)
        return collection.argmax(axis = 0)

def find_class_train(prediction):
    """
        Based on the value of the prediction in each of the cases, this fumction compares the columns of the matrix and assign as label the node with highest value.
    """
    predicted_class = [0] * len(prediction[0])

    collection = np.array(prediction)

    index = list(collection.argmax(axis = 0))

    for val in range(len(prediction[0])):
        predicted_class[val] = index[val] + 1
    return predicted_class

def unary_processing(train_set, author):
    """
        This method is used to convert the ground truth of a given author to 1 and other to 0 to perform a binary classification.
    """

    train = np.copy(train_set)
    for index in range(len(train)):
        if train[index][-1] != author:
            train[index][-1] = 0
        else:
            train[index][-1] = 1
    return list(train)
