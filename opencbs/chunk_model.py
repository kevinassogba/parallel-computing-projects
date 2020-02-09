import sys
import re
import operator
import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


dt_acurracy = 0
ada_accuracy = 0

def adaboost_classifier(X, Y):

    global dt_accuracy, ada_accuracy

    # Split dataset into training and inference data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada_model = ada_model.fit(X, Y)

    ada_accuracy = ada_model.score(X,Y)

    print("Accuracy = ", ada_accuracy)

    # with open('ada_model.obj','wb') as file:
    #     pickle.dump(adamodel, file)

    return ada_model


def decision_tree_classifier(X, Y):

    global dt_accuracy, ada_accuracy

    # Split dataset into training and inference data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    dt_model = tree.DecisionTreeClassifier()
    dt_model = dt_model.fit(X_train, y_train)

    predicted = dt_model.predict(X_test)

    dt_accuracy = accuracy_score(y_test, predicted)

    print("Accuracy = ", dt_accuracy)

    # with open('dt_model.obj','wb') as file:
    #     pickle.dump(dtmodel, file)

    return dt_model

def main():

    global dt_accuracy, ada_accuracy

    # Column names
    col_names = ['lead_zero', 'len_tar', 'comparison', 'comm_size', 'label']
    # load dataset
    dataset = pd.read_csv("data.csv", header=None, names=col_names)

    dataset.head()

    features = ['lead_zero', 'len_tar', 'comparison', 'comm_size']

    X = dataset[features]
    Y = dataset.label

    # 6,71,6,71,124,100000000
    Z = np.array([6, 71, 6, 124])
    Z = Z.reshape(1, -1)

    print('\nDecision tree classification')
    dt_model = decision_tree_classifier(X, Y)
    # with open('dt_model.obj', 'rb') as file:
    #     pred_model = pickle.load(file)
    #
    # predicted = pred_model.predict(Z)
    # print('Does the predicted result matches expectations? ', predicted == 100000000)
    # print("The predicted result is", predicted, 'while the expected is 100000000')

    print('\nAdaboost classification')
    ada_model = adaboost_classifier(X, Y)
    # with open('ada_model.obj', 'rb') as file:
    #     pred_model = pickle.load(file)
    #
    # predicted = pred_model.predict(Z)
    # print('Does the predicted result matches expectations? ', predicted == 100000000)
    # print("The predicted result is", predicted, 'while the expected is 100000000')

    print('\nComparing the accuracies to save the best model')
    with open('chunk_size_model.obj','wb') as file:
        if dt_accuracy > ada_accuracy:
                pickle.dump(dt_model, file)
        else:
                pickle.dump(ada_model, file)

    print('\nTesting the saved model')
    with open('chunk_size_model.obj', 'rb') as file:
        pred_model = pickle.load(file)

    predicted = pred_model.predict(Z)
    print('Does the predicted result matches expectations? ', predicted == 100000000)
    print("The predicted result is", predicted, 'while the expected is 100000000')


main()
