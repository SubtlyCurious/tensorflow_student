import tensorflow
import numpy
import sklearn
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.utils import shuffle

#Read csv file using pandas
data = pd.read_csv("student-mat.csv", sep=";")

#Trim data down
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #attributes

predict = "G3" #also known as a label

#Training data to predict another value
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#taking attributes and splitting them into four different arrays 
#the test are to test the accuracy of our model
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)