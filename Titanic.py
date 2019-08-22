import math
import numpy as nu 
import pandas as pd 
import seaborn as sns
from seaborn import countplot
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():

    #load data
    titanic_data = pd.read_csv("TitanicDataset.csv")

    #Analyze data
    figure()
    target = "Survived"
    countplot(data = titanic_data,x = target).set_title("Survived and non survived passangers")
    show()

    figure()
    target = "Survived"
    countplot(data = titanic_data,x = target,hue="Sex").set_title("Survived and non survived passangers on basis of gender")
    show()

    figure()
    target = "Survived"
    countplot(data = titanic_data,x = target,hue="Pclass").set_title("Survived and non survived passanger on basis of Passanger class")
    show()

    #Another way
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passanger on based on Age")
    show()

    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non survived passanger on based on Fare")
    show()

    #data cleaning
    titanic_data.drop("zero",axis = 1, inplace = True)

    x = titanic_data.drop("Survived",axis = 1)
    y = titanic_data["Survived"]

    #Training data
    xtrain, xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5)

    logmodel = LogisticRegression()
    logmodel.fit(xtrain,ytrain)

    #Testing data
    prediction = logmodel.predict(xtest)

def main():
    TitanicLogistic()

if __name__ == "__main__":
    main()


