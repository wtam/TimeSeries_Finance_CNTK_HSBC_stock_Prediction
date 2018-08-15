# A method which obtains stock data from Yahoo finance
# Requires that you have an internet connection to retreive stock data from Yahoo finance
import time
import datetime
import numpy as np
import pandas as pd
import os

from pandas_datareader import data as pdr

# My notes: otherwise wont' work as yahoo api depracated since early this yr
# These 2 lines only use when runing on DSVM
# import sys
# sys.path.append('/anaconda/envs/py35/lib/python3.5/site-packages/')

# pip install fix_yahoo_finance using the File->Open Command Prompt
import fix_yahoo_finance as fyf

fyf.pdr_override()


# issue fix https://github.com/ranaroussi/fix-yahoo-finance/issues/34

def get_stock_data(contract, s_year, s_month, s_day, e_year, e_month, e_day):
    """
    Args:
        contract (str): the name of the stock/etf
        s_year (int): start year for data
        s_month (int): start month
        s_day (int): start day
        e_year (int): end year
        e_month (int): end month
        e_day (int): end day
    Returns:
        Pandas Dataframe: Daily OHLCV bars
    """
    start = datetime.datetime(s_year, s_month, s_day)
    end = datetime.datetime(e_year, e_month, e_day)

    retry_cnt, max_num_retry = 0, 3

    while (retry_cnt < max_num_retry):
        try:
            bars = pdr.get_data_yahoo(contract, start, end)
            return bars
        except:
            retry_cnt += 1
            time.sleep(np.random.randint(1, 10))

    print("Yahoo Finance is not reachable")
    return None

import pickle as  pkl

# We search in cached stock data set with symbol SPY.
envvar = 'EXTERNAL_TESTDATA_SOURCE_DIRECTORY'
def is_test(): return envvar in os.environ

def download(data_file):
    data = get_stock_data('0005.HK', 2000,1,1, 2020,12,31) #My notes: set to the end date for end of  2020 to get whatever the last closing date
    dir = os.path.dirname(data_file)
    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(data_file):
        print("Saving", data_file )
        with open(data_file, 'wb') as f:
            pkl.dump(data, f, protocol = 2)
    return data

#On WorkBench the path is C:\Users\wdam\AppData\Local\Temp\azureml_runs\tensorflow-tutorial_1516675694122\data\Stock
data_file = os.path.join("data", "Stock", "stock_symbol.pkl")

# Check for data in local cache
if os.path.exists(data_file):
        print("File already exists", data_file)
        #data = pd.read_pickle(data_file)
        #still download the file to overwrite
        print("Removing the file and download again")
        os.remove(data_file)
        data = download(data_file)
else:
    # If not there we might be running in CNTK's test infrastructure
    if is_test():
        test_file = os.path.join(os.environ[envvar], 'data','Stock','stock_symbol.pkl')
        if os.path.isfile(test_file):
            print("Reading data from test data directory")
            data = pd.read_pickle(test_file)
        else:
            print("Test data directory missing file", test_file)
            print("Downloading data from Yahoo Finance")
            data = download(data_file)
    else:
        # Local cache is not present and not test env
        # download the data from Yahoo finance and cache it in a local directory
        # Please check if there is trade data for the chosen stock symbol during this period
        data = download(data_file)

#Swap the Volumne & Close col from the panda data frame using reindex
columnsTitles=["Open","High", "Low", "Volume", "Close", "Adj Close"]
data = data.reindex(columns=columnsTitles)

# drop the 'Adj Close' column
data.drop('Adj Close', axis=1, inplace=True)
data.tail(5)


import math
# pip install keras using the command prompt
####default keras backend is Tensorflow and to set CNTK, bee to edit c:\Users\wdam\.keras\keras.json
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import sklearn.preprocessing as prep
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
import tensorflow as tf
import horovod.keras as hvd
# Ref: source /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/bin/mpivars.sh; mpirun -n 2 -ppn 1 -hosts $AZ_BATCH_HOST_LIST -env I_MPI_DEBUG=6 -env I_MPI_FABRICS=tcp python $AZ_BATCHAI_INPUT_SCRIPTS/Horovod_StockPrediction-Keras.py 

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

batch_size = 1
epochs = 15
#num_classes = 1

# Horovod: adjust number of epochs based on number of GPUs.
###epochs = int(math.ceil(12.0 / hvd.size()))

# Based on this learning https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# Before we do anything, it is a good idea to fix the random number seed to ensure our results are reproducible.
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
# drop all colum and only remain the Closecolumn
dataframe = data
dataframe.drop('Open', axis=1, inplace=True)
dataframe.drop('High', axis=1, inplace=True)
dataframe.drop('Low', axis=1, inplace=True)
dataframe.drop('Volume', axis=1, inplace=True)
dataset = dataframe.values
# convert the dat from float64 to float32, otherwise, the training in each epoch will be slower due to expect inout is float32
dataset = dataset.astype('float32')

print ("4 day ago: ", dataset[len(dataset)-5])
print ("3 day ago: ", dataset[len(dataset)-4])
print ("2 day ago: ", dataset[len(dataset)-3])
print ("1 day ago: ",  dataset[len(dataset)-2])
print ("today-CurrentBid: ",dataset[len(dataset)-1])

# normalize the dataset, rescale the data to the range of 0-to-1
# We can easily normalize the dataset using the MinMaxScaler preprocessing class from the scikit-learn library
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print ("Normalized")
print ("4 day ago: ", dataset[len(dataset)-5])
print ("3 day ago: ", dataset[len(dataset)-4])
print ("2 day ago: ", dataset[len(dataset)-3])
print ("1 day ago: ",  dataset[len(dataset)-2])
print ("today: ",dataset[len(dataset)-1])

# split into train and test sets -> 7:3
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("length of dataset(total trading days): ", len(dataset), "\nlength of train dataset: ", len(train), "\nlength of test dataset: ", len(test))

# create a new dataset where X is the Closing Price of HSBC at a given time (t) and Y is the Closing Price at the next time (t + 1)
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    #for i in range(len(dataset)-look_back-1):
    for i in range(len(dataset)-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(testX[len(testX)-1])
print(testY[len(testY)-1])

# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of: [samples, time steps, features].
# Currently, our data is in the form: [samples, features]
print("Num of Training Sample(trainX): ", trainX.shape[0], "Num of feature(Closing Price): ", trainX.shape[1])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

testX[len(testX)-1]

# Determine how many batches are there in train and test sets
train_batches = len(trainX) // batch_size
test_batches = len(testX) // batch_size

# define thee LSTM network, 1 input, 4 hidden LSTM blocks or neurons and output layer that makes a single value prediction
# the network is trainned for 15 epochs and a batch size of 1 is used (E.g. for trainX sample size= 3137, each epoch will have 3137/1 iteraaction)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.add(Activation("linear"))

###model.compile(loss='mean_squared_error', optimizer='adam')
# Horovod: adjust learning rate based on number of GPUs (naive approach).
###opt = keras.optimizers.adam(lr=1.0 * hvd.size())
### since the accuracy is wrost with 70+ RMSE wiht the above horovod scale: 1.0 * hvd.size(), so let try back a much lower learning rate
opt = keras.optimizers.adam(0.001)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='mean_squared_error', optimizer='adam')

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

###model.fit(trainX, trainY, epochs=15, batch_size=1, verbose=2)
# Train the model.
model.fit(trainX, trainY, 
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs, 
            verbose=2,
            validation_data=(testX, testY))

score = model.evaluate(testX, testY, verbose=0)
print("Test score:", score)
print(model.summary())

#Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data.

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
##testPredict_y = model.predict(testY)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print("HSBC closing ", testX[len(testX)-4], testPredict[len(testX) -4])
print("HSBC closing ", testX[len(testX)-3], testPredict[len(testX) -3])
print("HSBC closing ", testX[len(testX)-2], testPredict[len(testX) -2])
print("HSBC closing ", testX[len(testX)-1], testPredict[len(testX) -1])






