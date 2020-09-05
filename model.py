import joblib

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

# define data import path
root = os.path.dirname(__file__)
path_df = os.path.join(root, 'data/cleaned_data.csv')
data = pd.read_csv(path_df)

scaler = MinMaxScaler()


# split data -- 75% will be training data, 25% will be test data
train, test = train_test_split(data, test_size=0.25)

X_train = train.drop('num', axis=1)
Y_train = train['num']

X_test = test.drop('num', axis=1)
Y_test = test['num']

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# begin Random Forest Classifier

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

# Testing model accuracy. Average is taken as test set is very small hence accuracy varies a lot
# everytime the model is trained

acc = 0
acc_binary = 0
for i in range(0, 20):
    Y_hat = rfc.predict(X_test)
    Y_hat_bin = Y_hat > 0
    Y_test_bin = Y_test > 0
    acc = acc + accuracy_score(Y_hat, Y_test)
    acc_binary = acc_binary + accuracy_score(Y_hat_bin, Y_test_bin)

print("Average test Accuracy:{}".format(acc/20))
print("Average binary accuracy:{}".format(acc_binary/20))

# Saving the trained model for inference
model_path = os.path.join(root, 'models/rfc.sav')
joblib.dump(rfc, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'models/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
