import sys
import pickle
import numpy as np

def main():
    input = sys.argv[1:]

    entry = np.array(input)
    entry = entry.reshape(1, -1)

    with open('chunk_size_model.obj', 'rb') as file:
        pred_model = pickle.load(file)

    predicted_size = pred_model.predict(entry)
    file.close()

    print(predicted_size[0])

main()
