
from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set_style('whitegrid')
%matplotlib inline




import os
os.chdir('F:\stocks')
list = os.listdir()
print(len(list))

df = pd.read_csv('F:\stocks\ibm.us.txt',sep=',')  ##ibm's..
df.head()

#filenames = random.sample([x for x in list ],10)  ## 10 randomm files
#df=pd.read_csv(filenames[0])
#df.head()


df[['Close']].plot()
plt.title("IBM closing..")
plt.show()


###train and test data

train = df[np.random.rand(len(df))<0.80] 
test=df[np.random.rand(len(df))<0.20]
print(len(df))

train.hist()
plt.show()



test.hist()
plt.show()



##neural net with ReLU as intermediate activation and linear output activatio
import tensorflow as tf


input_layer = tf.keras.layers.Input(shape=(10,))
layer1 = tf.keras.layers.Dense(units=64, activation='relu')(input_layer)
layer2 = tf.keras.layers.Dense(units=64, activation='relu')(layer1)
layer3 = tf.keras.layers.Dense(units=64, activation='relu')(layer2)
output = tf.keras.layers.Dense(units=1, activation='linear')(layer3)














