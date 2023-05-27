import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input

df = pd.read_csv('data/housepricedata.csv')

x = df.iloc[:, 0:10]
y = df.iloc[:, 10]

x = preprocessing.MinMaxScaler().fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

model = Sequential()
model.add(Input(shape=(10, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

test = [[8450, 7, 5, 856, 2, 1, 3, 8, 0, 548],             #1
        [14260, 8, 5, 1145, 2, 0, 3, 8, 1, 460],           #1
        [10084, 5, 5, 796, 1, 1, 2, 6, 0, 608],            #1
        [6120, 7, 8, 832, 1, 0, 2, 5, 0, 576],             #0
        [12968, 5, 6, 912, 1, 0, 2, 4, 0, 352],            #1
        [5000, 5, 5, 1340, 1, 1, 2, 6, 0, 205]]            #0

test = preprocessing.MinMaxScaler().fit_transform(test)
print(model.predict(test))

model.save('model.h5')
