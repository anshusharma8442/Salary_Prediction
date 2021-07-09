import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle


#data gathering from the csv file
df = pd.read_csv('hiring.csv')

#filling the null values
df['experience'].fillna(0, inplace=True)

df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(), inplace =True)

#data  splitting
X = df.iloc[:,:3]

#converting string number to integer
def string_to_number(word):
    dict ={'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
         'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 0:0}
    return dict[word]

df['experience'] = df['experience'].apply(lambda x: string_to_number(x))

Y = df.iloc[:,-1]

#splitting and training machine
from sklearn.linear_model import LineraRegression
regressor = LineraRegression()

#fitting data 
regressor.fit(X,Y)

#saving model to disc
pickle.dump(regressor, open('salary_prediction.pkl', 'wb'))

#loading model to compare the result
model = pickle.load(open('salary_prediction.pkl', 'rb'))
print(model.predict([[2,9,6]]))