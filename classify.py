import sys
import re
import operator
import pickle
import dt_class
import log_class
from collections import Counter

class initialization:
    """
        This class is designed to process the file and compute
        the value corresponding to each feature.
    """
    __Slots__: "data_frame"; "num_obs"; "task"; "ground_truth"


    def process(self, file, type):
        """
            process takes a file and stores its content into a buffer.
        """

        if type == "txt":
            file = open(file, encoding = 'utf-8')
        return self.fill_buffer(file)

    def fill_buffer(self, file):

        buffer = []
        records = []

        for line in file:
            words = re.findall(r"[\w']+|[.,!?;\"]", line)
            for  word in words:
                if len(buffer) < 300:
                    buffer.append(word.lower())
                else:
                    records.append(buffer)
                    buffer = []
                    buffer.append(word)

        return records

    def feature_engineering(self,records, filename):

        """
            feature_engineering takes each portion of text and returns
            an array of the values of the features which are considered
        """
        recordData = []
        recordData = self.punctuation_features(records, recordData)
        recordData = self.conjunction_features(records, recordData)
        recordData = self.pronoun_features(records, recordData)
        recordData = self.other_features(records, recordData)

        return recordData

    def punctuation_features(self, records, recordData):

        # The number of quotes = (number of ") / 2
        quotes = records.count('\"') // 2
        # recordData.append(quotes)

        # The number of "," : comma = records.count(',')
        comma = records.count(',')
        # recordData.append(comma)

        # The number of ";" : semiColumn = records.count(';')
        semiColumn = records.count(';')
        recordData.append(semiColumn)

        # The number of ":" : Column = records.count(':')
        column = records.count(':')
        # recordData.append(column)

        # number of sentences : sentence = records.count('.')
        sentence = records.count('.')
        # recordData.append(sentence)

        # recordData.append(quotes + column + semiColumn)
        recordData.append(comma + sentence)
        # recordData.append(quotes + comma + semiColumn + column + sentence)

        return recordData

    def conjunction_features(self, records, recordData):

        # Number of coordinating conjunction
        coordinating = records.count('for') + records.count('and') +\
            records.count('nor') + records.count('but') + records.count('or')+\
            records.count('yet') + records.count('so')
        # recordData.append(coordinating)

        # Number of subordinating conjunction
        subordinating = records.count('after') + records.count('although') +\
            records.count('unless') + records.count('because') + \
            records.count('since') + records.count('if') + records.count('that')
        # recordData.append(subordinating)

        return recordData

    def pronoun_features(self, records, recordData):

        # The number of instances of first person pronoun: I, me, my, mine
        first = records.count('i') + records.count('me') +\
                                    records.count('my') + records.count('mine')
        # recordData.append(first)
        # Third person
        third = records.count('she') + records.count('he') \
                                                        + records.count('they')
        recordData.append(third)

        # Use of the
        the = records.count('the')
        recordData.append(the)

        return recordData

    def other_features(self, records, recordData):

        words = []
        for element in records:
            if element not in ['.',',','!','?',';','"']:
                words.append(element)

        # Sentence length = number of words / number of sentences
        sentence = records.count('.')
        wordBysentence = round(len(words)/(sentence if sentence > 0 else 1))
        recordData.append(wordBysentence)

        # Count of the most repeated word
        most_words = Counter(words).most_common(5)
        lengths = 0
        counts = 0
        for element in most_words:
            if element[1] > 2:
                lengths += len(element[0])
                counts += element[1]
        recordData.append(lengths)

        # Average length of words and Maximum word length
        total = 0
        max = 0
        for word in words:
            wordlen = len(word)
            if wordlen > max:
                max = wordlen
            total += wordlen
        recordData.append((total // len(words)))
        recordData.append(max)

        return recordData

class classifiers:

    __Slots__: 'data_frame'; 'X_train'; 'X_test'; 'y_test'; 'y_train';'dt_model'; \
                            'ground_truth'; 'authors'; 'best_model'

    def __init__(self, task = ''):

        self.authors = {0: 'Arthur Conan Doyle',
                        1: 'Herman Melville',
                        2: 'Jane Austen'}

    def predict(self, data, model):
        """
            prediction tool which used fit to check for validity.
        """
        if isinstance(model, list):
            predicted_y  = [0]* len(model)
            for index in range(len(model)):
                predicted_y[index] = log_class._sigmoid(data, model[index])
            author = log_class.get_class(predicted_y)
            statement = 'Logistic Regression | Author: '+ self.authors[author]
        else:
            label = dt_class.fit(data, model)
            author = max(label.items(), key=operator.itemgetter(1))[0]
            statement = 'Decision Tree | Author: ' + self.authors[author - 1]
        return statement

def main(user_entry, selection):

    """
        This function contains the main commands to perform the tasks.
        In case of prediction, the file is preprocessed based on the best model,
        and the prediction is returned. In cae of training, the entire process
        takes place.
    """
    type = "text"

    if re.match("^[.txt]", user_entry):
        print("OK")
        type = "txt"

    filename = user_entry
    data = initialization()
    record = data.process(filename, type)
    prep_data = data.feature_engineering(record[0], filename)

    my_classifier = classifiers()

    with open('dt_model.obj', 'rb') as file:
        dt_model = pickle.load(file)

    with open('log_model.obj', 'rb') as file:
        log_model = pickle.load(file)

    with open('best_model.obj', 'rb') as file:
        best_model = pickle.load(file)

    if selection == 1:
        return my_classifier.predict(prep_data, dt_model)
    elif selection == 2:
        return my_classifier.predict(prep_data, log_model)
    elif selection == 3:
        return my_classifier.predict(prep_data, dt_model) + "\n\n" +       my_classifier.predict(prep_data, log_model)
    elif selection == 4:
        return my_classifier.predict(prep_data, best_model)
