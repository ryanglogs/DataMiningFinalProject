# plt.figure(figsize=(15,5))
# plt.plot(mainDF['Open'].values, color = 'red', label = 'open')
# plt.plot(mainDF['High'].values, color = 'black', label = 'high')
# plt.plot(mainDF['Low'].values, color = 'blue', label = 'low')
# plt.plot(mainDF['Close'].values, color = 'green', label = 'close')
# plt.title('Stock price')
# plt.xlabel('time (days)')
# plt.ylabel('price')
# plt.legend(loc='best')
# plt.show()
# print('x_train.shape = ',x_train.shape)
# print('y_train.shape = ', y_train.shape)
# print('x_valid.shape = ',x_valid.shape)
# print('y_valid.shape = ', y_valid.shape)
# print('x_test.shape = ', x_test.shape)
# print('y_test.shape = ',y_test.shape)

# plt.figure(figsize=(15,5))
# plt.plot(normalizedDF['Open'].values, color = 'red', label = 'open')
# plt.plot(normalizedDF['High'].values, color = 'black', label = 'high')
# plt.plot(normalizedDF['Low'].values, color = 'blue', label = 'low')
# plt.plot(normalizedDF['Close'].values, color = 'green', label = 'close')
# plt.title('Stock price')
# plt.xlabel('time (days)')
# plt.ylabel('normalized price')
# plt.legend(loc='best')
# plt.show()
#
# close = pd.Series(data=df['Close'].values, index=df['Date'])
# close.plot(figsize=(16,4), label="close Prices", legend=False)
# plt.show()


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import os
import time

valid_set_size_percentage = 20
test_set_size_percentage = 20

#step 1: convert numeric data into value between (0,1) using
#	newvalue = (original - minimum) / (maximum - minimum)
#	for each column in the dataframe
def normalize_data(df):
	for column in df:
		series = df[column]
		min = round(float(df[column].min()), 4)
		max = round(float(df[column].max()), 4)
		#print(column, min, max)
		temp = []
		for value in series:
			newValue = (value - min) / (max - min)
			temp.append(newValue)
		temp = pd.DataFrame(temp)
		df[column] = temp
		#modifiedDF=pd.concat([modifiedDF, temp], axis=1)
	# modifiedDF.columns = ['Open', 'High', 'Low', 'Close']
	# print(modifiedDF.head())
	return df

#this method splits the original dataset into three parts:
#	training dataset, used to train the model 
#	validation dataset, used to validate the models predictions with the real values 
#	test dataset, uses the model created to predict a certain number of values 
def create_sets(norm_data, seq_length):
	raw_data = norm_data.to_numpy()
	data = []
	
	for index in range(len(raw_data) - seq_len):
		data.append(raw_data[index: index + seq_len])
		
	data = np.array(data)
	valid_set_size=int(np.round(valid_set_size_percentage/100*data.shape[0]))
	test_set_size=int(np.round(test_set_size_percentage/100*data.shape[0]))
	train_set_size = data.shape[0] - (valid_set_size + test_set_size)
	
	x_train=data[:train_set_size, :-1, :]
	y_train=data[:train_set_size, -1, :]
	
	x_valid=data[train_set_size:train_set_size+valid_set_size, :-1, :]
	y_valid=data[train_set_size:train_set_size+valid_set_size, -1, :]
	
	x_test=data[train_set_size+valid_set_size:, :-1, :]
	y_test=data[train_set_size+valid_set_size:, -1, :]
	
	return [x_train, y_train, x_valid, y_valid, x_test, y_test]

#
#
#
#
def get_next_batch(batch_size):
	global index_in_epoch, x_train, perm_array
	start = index_in_epoch
	index_in_epoch += batch_size
	
	if index_in_epoch>x_train.shape[0]:
		np.random.shuffle(perm_array) # shuffle permuation array
		start =0
		index_in_epoch=batch_size
		
	end = index_in_epoch
	
	#print ('Perm array dtype',perm_array.dtype)
	return x_train[perm_array[start:end]], y_train[perm_array[start:end]]




data = open("ge.us.csv", "r")
mainDF = pd.read_csv(data)
mainDF = mainDF.drop(columns='OpenInt') #openint is not useful in this dataset
mainDF = mainDF.drop(columns='Volume') #volume is also not useful in this dataset 
dateDF = mainDF['Date']
mainDF=mainDF.drop(columns='Date') #we import date incase we need it later to plot and then drop from the main dataframe


normalizedDF = mainDF.copy()
normalizedDF = normalize_data(normalizedDF)

seq_len = 20
x_train, y_train, x_valid, y_valid, x_test, y_test = create_sets(normalizedDF, seq_len)

# plt.figure(figsize=(15,5))
# plt.plot(normalizedDF['Open'].values, color = 'red', label = 'open')
# plt.plot(normalizedDF['High'].values, color = 'black', label = 'high')
# plt.plot(normalizedDF['Low'].values, color = 'blue', label = 'low')
# plt.plot(normalizedDF['Close'].values, color = 'green', label = 'close')
# plt.title('Stock price')
# plt.xlabel('time (days)')
# plt.ylabel('normalized price')
# plt.legend(loc='best')
# plt.show()

#print('x_train.shape[0] = ', x_train.shape[0])
#print(x_train[0])

index_in_epoch=0
perm_array=np.arange(x_train.shape[0])

np.random.shuffle(perm_array)

t0 = time.time()

# parameters
n_steps = seq_len-1
n_inputs=4
n_neurons=200
n_outputs=4
n_layers=2
learning_rate=0.001
batch_size=75
n_epochs=100
train_set_size=x_train.shape[0]
test_set_size=x_test.shape[0]

print('\n Train set size {0}; Test set size {1} \n'.format(train_set_size, test_set_size))

tf.compat.v1.reset_default_graph()

X = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.compat.v1.placeholder(tf.float32, [None, n_outputs])

layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.elu)
			for layer in range(n_layers)]

multi_layer_cell=tf.compat.v1.nn.rnn_cell.MultiRNNCell(layers)
rnn_outputs, states = tf.compat.v1.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.compat.v1.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:, n_steps-1, :]

loss = tf.reduce_mean(input_tensor=tf.square(outputs-Y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

sumTrain = 0
sumValid = 0

with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	for iteration in range(int(n_epochs*train_set_size/batch_size)):
		x_batch, y_batch = get_next_batch(batch_size)
		sess.run(training_op, feed_dict={X: x_batch, Y: y_batch})
		if iteration % int(5*train_set_size/batch_size) == 0:
			mse_train = loss.eval(feed_dict={X: x_train, Y: y_train})
			mse_valid = loss.eval(feed_dict={X: x_valid, Y: y_valid})
			print('{0} epochs: MSE train/valid = {1}/{2}'.format(iteration*batch_size/train_set_size, mse_train, mse_valid))
			sumTrain+=mse_train
			sumValid+=mse_valid
			
	y_train_pred = sess.run(outputs, feed_dict={X: x_train})
	y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
	y_test_pred = sess.run(outputs, feed_dict={X: x_test})


avgTrain = sumTrain / 20
avgValid = sumValid / 20
print("Time elapsed: ", ((time.time()-t0)/60))
print("Avg MSE train/valid", avgTrain, avgValid)
print("Batch size / number of epochs", batch_size, n_epochs)

ft=0 #0=open, 1=close, 2=high, 3=low

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')
plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft], color='gray', label='valid target')
plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0], y_train.shape[0]+y_test.shape[0]+y_test.shape[0]), y_test[:,ft], color='black', label='test target')
plt.plot(np.arange(y_train_pred.shape[0]), y_train_pred[:,ft], color='red', label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]), y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]), y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time (days)')
plt.ylabel('normalized prices')
plt.legend(loc='best')

plt.subplot(1,2,2);

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]), y_test[:,ft], color='black', label='test target')
plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]), y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time (days)')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.show()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
