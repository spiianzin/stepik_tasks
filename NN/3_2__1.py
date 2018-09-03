import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l) 2,3 
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    
    print(deltas)
    print(sums)
    print(weights)
    print('===================================')
    
    # here goes your code
    A = weights.T.dot(deltas.T)
    print(A)
    B = sigmoid_prime(sums)
    print(B)
    
    print(A.shape)
    print(B.shape)
    C = A.T * B
    print(C)
    D = C.mean(axis=0)
    print(D)
    print(D.shape)
    
    return ((weights.T.dot(deltas.T)).T * sigmoid_prime(sums)).mean(axis=0)
    
    
deltas = np.array([[0.3, 0.2], [0.3, 0.2]])
sums = np.array([[0, 1, 1], [0, 2, 2]])
weights = np.array([[0.7, 0.2, 0.7], [0.8, 0.3, 0.6]])
        
print get_error(deltas, sums, weights)
