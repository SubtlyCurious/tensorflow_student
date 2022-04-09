import tensorflow
import numpy
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle

from matplotlib import style
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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

'''
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test,y_test)
    #Acutal model that predicts scores at the end of the year
    print(accuracy)

    #finding the best model
    if accuracy > best:
        best = accuracy
        #Storing model
        with open('studentmodel.pickle','wb') as f:
            pickle.dump(linear, f)

'''

#reading model file
pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

#print line coefficients
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Array of Arrays
predictions = linear.predict(x_test)

#print out all the predictions and display input data 
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#plotting data
p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()