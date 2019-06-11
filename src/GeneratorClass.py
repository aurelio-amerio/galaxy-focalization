import keras
import numpy as np

from keras.layers import Lambda, Dense, Input, Layer, Reshape
from keras.models import Model
from keras.models import Sequential
from keras import backend as K


class DataGeneratorEncoders_MLP(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, get_data_fn, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 shuffle=True):
        """Initialization"""
        self.get_data_fn = get_data_fn
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_train, X_target = self.__data_generation(list_IDs_temp)

        return X_train, X_target

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X_train = np.empty((self.batch_size, *self.dim, self.n_channels))
        # X_target = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_train_tmp = np.empty((self.batch_size, *self.dim))
        X_target_tmp = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X_train_tmp[i,], X_target_tmp[i,] = self.get_data_fn(ID)

        X_train = X_train_tmp.reshape((-1, self.dim[0] * self.dim[1]))
        X_target = X_target_tmp.reshape((-1, self.dim[0] * self.dim[1]))
        return X_train, X_target




class DataGeneratorEncoders_CNN(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, get_data_fn, batch_size=32, dim=(32, 32), n_channels=1,
                 shuffle=True):
        """Initialization"""
        self.get_data_fn = get_data_fn
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_train, X_target = self.__data_generation(list_IDs_temp)

        return X_train, X_target

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X_train = np.empty((self.batch_size, *self.dim, self.n_channels))
        # X_target = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_train = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_target = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            tmp1, tmp2 = self.get_data_fn(ID)
            X_train[i,] = tmp1.reshape(
                (*self.dim, 1))  # + 0.001 * np.random.random(size=self.dim).reshape((*self.dim, 1))
            X_target[i,] = tmp2.reshape((*self.dim, 1))

        return X_train, X_target

    # In[7]:
