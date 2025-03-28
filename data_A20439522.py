import csv
import math

def data(csv_file):
    training_data = {}
    test_data = {}
    rows = []
    with open(csv_file) as file_handle:
        file_reader = csv.reader(file_handle)
        file_reader.__next__()
        [rows.append(row) for row in file_reader]
        learning_data = math.ceil(len(rows) * .8)
        for i in range(0, learning_data):
            review = rows[i]
            training_data[i+1] = [review[1], review[4]]
        for i in range(learning_data, len(rows)):
            review = rows[i]
            test_data[i+1] = [review[1], review[4]]
    return training_data, test_data
