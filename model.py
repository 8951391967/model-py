import torch
import numpy as np 
import pandas as pd
import torch.nn as nn
import joblib
import sklearn
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split



# data taking


data  = pd.read_csv("diabetes.csv")


# data sorting, remove the zero in independent vari and standerdise them 
# if we call .to_numpy() functiion then the text will erase so we have to call again


x = data.drop(columns = ["Outcome"]).to_numpy()
y = data['Outcome'].to_numpy()


# np.where(condition, value_if_true, value_if_false)
# get_loc()  used to the get the index of row or soloums you wanted



colum_zeros =  ['Glucose', 'Insulin', 'BloodPressure', 'SkinThickness', 'BMI']
for colum in colum_zeros:
    x[ : ,  data.columns.get_loc(colum)] = np.where( x[ :, data.columns.get_loc(colum)] == 0 ,
                                                     np.nan ,
                                                       x[ : , data.columns.get_loc(colum)])
    col_index = data.columns.get_loc(colum)
    col_median = np.nanmedian(x[: , col_index])
    x[:, col_index] = np.where(np.isnan(x[:, col_index]), col_median, x[:, col_index])

# data dividing ( train and test)


#i = len(x)
#length = int(i * 0.8)


#x_train , y_train = x[ : length] , y[ : length]
#x_test , y_test = x[length : ] , y[length: ]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)
  
print(x_train.shape)
print(y_train.shape)

 # shape (m,1)

# to visulaise the data



"""plt.scatter(x_train[:, 0] c= 'blue')
plt.xlabel(" independent value")
plt.ylabel("dependent value")
plt.title("the train grapgh")
plt.show()"""

# standardise the data bec large will always dominate  


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
# model building


class linearregression():

    def __init__(self, lr = 0.01 , epochs = 1000):

        self.lr = 0.01
        self.epochs = 1000
        self.w = 0
        self.b = 0
# activation         

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    


    def fit_train(self,x_train,y_train):
        m ,n = x_train.shape
        y_train = y_train.reshape(-1,1) 
        self.w = np.random.randn(n, 1) * 0.01
        self.b = 0

        self.b = 0
          # shape (m,1)


        
        for epoch in range(self.epochs):

# fromula
            z = np.dot(x_train, self.w) + self.b
            y_hat = self.sigmoid(z)


# gradient
            dw = (1/m)* np.dot(x_train.T, (y_hat - y_train)) 
            db =  (1/m)*np.sum(y_hat -y_train)

# update 

             
            self.w =self.lr * dw
            self.b =self.lr * db

# why losss function not needed simply to see the how it got 

        loss = - (1/m) * np.sum(y_train * np.log(y_hat + 1e-8) + (1-y_train) * np.log(1 - y_hat + 1e-8))
        print("net loss" , loss) 

          


# initalisng
    def pridict_prob(self , X):
            z = np.dot(X, self.w) + self.b
            probs_class1 = self.sigmoid(z)         # probability of class 1
            probs_class0 = 1 - probs_class1        # probability of class 0
            return np.c_[probs_class0, probs_class1]
           
            

    def probict(self ,X,  threshold= 0.5):
            probabilities = self.pridict_prob(X)  # make sure this returns probabilities
            return (probabilities >= threshold).astype(int)
            


model = linearregression()
model.fit_train(x_train, y_train)   

y_pred_test = model.probict(x_test)
y_prob_test = model.pridict_prob(x_test)

print(y_pred_test.shape)
print(x_test.shape)
print(y_train.shape)

accuracy = np.mean(y_pred_test  ==  y_test)
print(f"the acccuracy of model = {accuracy}")

joblib.dump(model, "logistic_model.joblib")
joblib.dump(scaler, "scaler.joblib")


