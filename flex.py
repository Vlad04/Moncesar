import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import math
import operator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

failed_components_list = []

class Node(object):
    def __init__(self, component):
        self.component_name = component
        self.occurences = 1

    def increase_occurence(self):
        self.occurences += 1

class TrainingData:
    
    def training_data_loader(self):
        data = pd.read_csv("4_NNTMRT0P1CPK.csv")
        X_dataset = data.iloc[:, 1:8].values
        y_dataset = data.iloc[:, 8].values

        scaler = StandardScaler()
        scaler.fit(X_dataset)

        classifier = KNeighborsClassifier(n_neighbors=15)

        classifier.fit(X_dataset, y_dataset)

        return  classifier

    def append_error_node(self, node_array, component):
        for tmp in node_array:
            if tmp.component == component:
                tmp.increase_occurence(self)
            else:
                node_array.append(Node(component))

    def predict_new_row(self, dataset_row, classifier):
        prediction = classifier.predict(dataset_row)
        return prediction

    def do_classification(self, file, classifier):
        scaler = StandardScaler()
        data = []
        counter = 0
        error_list = []
        with open(file) as csv_file:
            next(csv_file, None)
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                #data.append([float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14])])
                print(row[8])
                height = float(row[8])
                area = float(row[9])
                area_percentage = float(row[10])
                volume = float(row[11])
                volume_percentage = float(row[12])
                offsetX = float(row[13])
                offsetY = float(row[14])
                data.append([height, area, area_percentage, volume, volume_percentage, offsetX, offsetY])

                #data.append([row[8], row[9], row[10], row[11], row[12], row[13], row[14]])
                scaler.fit(data)
                result = self.predict_new_row(data, classifier)
                if(result == 0):
                    
                    error_row  = "Component Number: " + str(row[5])+ " Area(%): " + str(area_percentage) +" Volume(%): " + str(volume_percentage) +" OffsetX: " + str(offsetX) +" OffsetY: " + str(offsetY)
                    error_list.append(error_row)
                    #print("Error on:") #5 ComponentNumber, 6 PinNumber
                    #print("Component Number -> ", row[5], ", Pin Number -> ", row[6])
                    counter += 1
                data = [] # Reset data
            print("Total Errors: ", counter)
            return error_list





