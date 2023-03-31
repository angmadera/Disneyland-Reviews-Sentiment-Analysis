import csv
import math

def training_data(csv_file):
    training_data = {}
    rows = []
    with open(csv_file) as file_handle:
        file_reader = csv.reader(file_handle)
        file_reader.__next__()
        [rows.append(row) for row in file_reader]
        learning_data = math.ceil(len(rows) * .8)
        for i in range(0, learning_data):
            review = rows[i]
            training_data[i+1] = [review[1], review[4]]
    return training_data

# def training_data(csv_file):
#     training_data = {}
#     rows = []
#     with open(csv_file) as file_handle:
#         file_reader = csv.reader(file_handle)
#         file_reader.__next__()
#         [rows.append(row) for row in file_reader]
#         learning_data = math.ceil(len(rows) * .8)
#         for i in range(learning_data + 1, ):
#             review = rows[i]
#             training_data[i+1] = review[4]
#     return training_data