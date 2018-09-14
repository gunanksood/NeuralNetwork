from __future__ import print_function
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as r
from sklearn.metrics import accuracy_score


digits = load_digits()
print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[3])
plt.show()
# shows the number pixels

print(digits.data)

X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
print(X[0, :])

y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
print(y_train[0], y_v_train[0])

nn_structure = [64, 30, 10]


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    w = {}
    b = {}
    for l in range(1, len(nn_structure)):
        w[l] = r.random_sample((nn_structure[l], nn_structure[l - 1]))
        b[l] = r.random_sample((nn_structure[l],))
    return w, b


def init_tri_values(nn_structure):
    tri_w = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_w[l] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        tri_b[l] = np.zeros((nn_structure[l],))

    return tri_w, tri_b


def feed_forward(x, w, b):
    h = {1: x}
    z = {}

    for l in range(1, len(w) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]

        z[l + 1] = w[l].dot(node_in) + b[l]
        h[l + 1] = f(z[l + 1])

    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    return -(y - h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_l, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_l) * f_deriv(z_l)


def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    w, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient for  {} iterations '.format(iter_num))
    while cnt < iter_num:
        if cnt % 1000 == 0:
            print("iterations {} of {}".format(cnt, iter_num))
        tri_w, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            h, z = feed_forward(X[i, :], w, b)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i, :], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i, :] - h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], w[l], z[l])
                    tri_w[l] += np.dot(delta[l + 1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    tri_b[l] += delta[l + 1]

        for l in range(len(nn_structure) - 1, 0, -1):
            w[l] += -alpha * (1.0 / m * tri_w[l])
            b[l] += -alpha * (1.0 / m * tri_b[l])

        avg_cost = 1.0 / m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return w, b, avg_cost_func


w, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)
plt.plot(avg_cost_func)
plt.ylabel('Average j')
plt.xlabel('Iteration Number')
plt.show()


def predict_y(w, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], w, b)
        y[i] = np.argmax(h[n_layers])
    return y


y_pred = predict_y(w, b, X_test, 3)
print (accuracy_score(y_test, y_pred) * 100)

