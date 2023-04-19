import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras import layers, losses
from sklearn.metrics import classification_report

df = pd.read_csv('creditcard.csv')

#data preparation
X, y = df.drop('Class', axis=1), df.Class
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#building autoencoder model
X_train_normal = X_train[np.where(y_train == 0)]
inputs = tf.keras.layers.Input(shape=(30, ))
encoder = tf.keras.Sequential([
    layers.Dense(5, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(3, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(1, activation='relu')])(inputs)

decoder = tf.keras.Sequential([
    layers.Dense(1, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(3, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(30, activation='sigmoid')])(encoder)
autoencoder = tf.keras.Model(inputs=inputs, outputs=decoder)

#compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')

#fit the autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal,
                          epochs=20,
                          batch_size=64,
                          validation_data=(X_test, X_test),
                          shuffle=True)

def plot_loss(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title('Loss Curve Plot')
    plt.legend()
    plt.show()
plot_loss(history)

#predict anomalies
prediction = autoencoder.predict(X_test)
#get the mean absolute error 
prediction_loss = tf.keras.losses.mae(prediction, X_test)
loss_threshold = np.percentile(prediction_loss, 95)
print(f'The prediction loss threshold for 5% of outliers is {loss_threshold: .2f}')

#visualize the threshold
sns.histplot(prediction_loss, bins=30, alpha=0.8)
plt.axvline(x=loss_threshold, color='orange')
plt.show()

#checking model performance
threshold_prediction = [0 if i < loss_threshold else 1 for i in prediction_loss]
#check prediction performance
print(classification_report(y_test, threshold_prediction))
