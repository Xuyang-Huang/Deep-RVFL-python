#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Deep_RVFL.py
#@Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset


num_nodes = 50  # Number of enhancement nodes.
regular_para = 0.01  # Regularization parameter.
weight_random_range = [-1, 1]  # Range of random weights.
bias_random_range = [-1, 1]  # Range of random weights.
num_layer = 4  # Number of hidden layers


class DeepRVFL:
    """A deep RVFL classifier.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        n_layer: A integer, N=number of hidden layers.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
    """
    def __init__(self, n_nodes, lam, w_random_vec_range, b_random_vec_range, activation, n_layer, same_feature=False):
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature

    def train(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        n_feature = len(data[0])
        d = self.standardize(data, 0)  # Normalization data
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.b_random_range))
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, d], axis=1)

        y = self.one_hot(label, n_class)
        if n_sample > (self.n_nodes * self.n_layer + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(n_feature + self.n_nodes * self.n_layer) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """
        n_sample = len(data)
        d = self.standardize(data, 0)
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, d], axis=1)
        result = np.dot(d, self.beta)
        if not output_prob:
            result = np.argmax(result, axis=1)
        return result

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: Accuracy.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        n_sample = len(data)
        d = self.standardize(data, 0)
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, d], axis=1)
        result = np.dot(d, self.beta)
        result = np.argmax(result, axis=1)
        acc = np.sum(np.equal(result, label))/len(label)
        return acc

    def get_random_vectors(self, m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    def one_hot(self, x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.std(x)
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.std(x, axis=0)
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]


class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def sine(self, x):
        return np.sin(x)

    def hardlim(self, x):
        return (np.sign(x) + 1) / 2

    def tribas(self, x):
        return np.maximum(1 - np.abs(x), 0)

    def radbas(self, x):
        return np.exp(-(x**2))

    def sign(self, x):
        return np.sign(x)

    def relu(self, x):
        return np.maximum(0, x)


def prepare_data(proportion):
    dataset = sk_dataset.load_wine()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    train, val, num_class = prepare_data(0.9)
    deep_rvfl = DeepRVFL(num_nodes, regular_para, weight_random_range, bias_random_range, 'relu', num_layer, False)
    deep_rvfl.train(train[0], train[1], num_class)
    prediction = deep_rvfl.predict(val[0], output_prob=False)
    accuracy = deep_rvfl.eval(val[0], val[1])
