import myanfis
import pandas as pd
import tensorflow as tf
import numpy
import sys
from keras.models import load_model

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#read the data from the csv file
data = pd.read_csv("C:\\Users\\Hari\\Downloads\\dataset.csv")

#split the data as train data and test data
#X, Y are train data
#x, y ara test data

standardScaler = StandardScaler()
dataset = data.copy()
dataset["Dummy"] = standardScaler.fit_transform((data[["Dummy"]]))
dataset["RSSI"] = standardScaler.fit_transform(data[["RSSI"]])
dataset["Distance"] = standardScaler.fit_transform(data[["Distance"]])

X = dataset.iloc[:30, [0, 1]]
Y = dataset.iloc[:30, [-1]]

x = dataset.iloc[30:, [0, 1]]
y = dataset.iloc[30:, [-1]]

print(x)
print(y)
param = myanfis.fis_parameters(
    n_input=2,
    n_memb=2,
    batch_size=5,
    memb_func='gaussian',
    optimizer='sgd',
    loss=tf.keras.losses.MeanAbsoluteError(),
    n_epochs=30
)

kfold = KFold(n_splits=2)
histories = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    fis = myanfis.ANFIS(n_input=param.n_input,
                        n_memb=param.n_memb,
                        batch_size=param.batch_size,
                        memb_func=param.memb_func,
                        name="firstAnfis")
    fis.model.compile(
        optimizer=param.optimizer,
        loss=param.loss,
        metrics=['mae']
    )

    history = fis.fit(X_train, Y_train,
                      epochs=param.n_epochs,
                      batch_size = param.batch_size,
                      validation_data = (X_test, Y_test)
                      )

    histories.append(history)


fis.model.save("predict.h5")

