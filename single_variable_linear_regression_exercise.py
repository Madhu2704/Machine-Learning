
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pprint import pprint


def plotTheGraph(df, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(df[xlabel], df[ylabel], color='red', marker="*")
    plt.show()


def plotTheGraphWithSlopeLine(df, regression, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(df[xlabel], df[ylabel], color='red', marker="*")
    plt.plot(df[xlabel], regression.predict(
        df[['year']]), color='blue')
    plt.show()


def regressor(df, xlabel, ylabel):
    regression = LinearRegression()
    regression.fit(df[[xlabel]],
                   df[ylabel])
    return regression


CANADA_INCOME_DATASET = "canada_per_capita_income.csv"
canada_income_dataset = pd.read_csv(CANADA_INCOME_DATASET)
print(canada_income_dataset)


# plotTheGraph(canada_income_dataset, 'year','per capita income (US$)')

LR = regressor(canada_income_dataset, 'year',
               'per capita income (US$)')
print(LR.predict([[int(input("Enter the Year To get the Salary:"))]]))

# plotTheGraphWithSlopeLine(canada_income_dataset,
#                           LR, 'year', 'per capita income (US$)')


X = np.array(canada_income_dataset['year']).reshape(-1, 1)
y = np.array(canada_income_dataset['per capita income (US$)']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regression = LinearRegression()
regression.fit(X_train,
               y_train)

print(regression.score(X_test, y_test))
