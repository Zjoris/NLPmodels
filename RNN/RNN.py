'''
This file contains an implementation of a Recurrent Neural Network 
built from scratch using numpy, ... etc



Notes to self or to do list: 
   - implement backpropagation
   - Implement deep hidden layer (weight initialization)
   - Add bias? 
   - Fix Y matrix in RNN_forward (seems a bit inefficient)
'''
import numpy as np
import math

from numpy.lib.shape_base import expand_dims


# A function to calculate the sigmoid of X (single value or array)
def sigmoid_func(X) :
    return 1/(1+np.exp(-X))

# A function to calculate the softmax of X (single value or array)
def softmax_func(X) :
    return np.exp(X)/np.sum(np.exp(X))

# A function to calculate the tanh of X (single value or array)
def tanh_func(X) :
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))



# Recurrent Neural Network
class RNN() :
    def __init__(self) :
        self._U = None
        self._W = None
        self._V = None
        self._D = None

    # Initialize random weights where 
    #      U is the weight matrix from input to hidden layers
    #      W is the weight matrix from hidden layer to hidden layer
    #      V is the weight matrix from hidden layer to output
    # 
    #  If I want the hidden layer to be able to be a deep Neural Net
    #  I will need extra weight matrices for that neural net
    #
    #  For now I will keep it simple with one recurrent hidden layer
    def init_weights(self,shapes = (1,(100,),1)) :
        # What about bias? 
        self._U = np.random.uniform(-0.3,0.3,(shapes[0],shapes[1][0]))
        self._W = np.random.uniform(-0.3,0.3,(shapes[1][0],shapes[1][-1]))
        self._V = np.random.uniform(-0.3,0.3,(shapes[1][-1],shapes[2]))
        #self._D = n_layer_init(shapes[1])

    def one_layer_out(self,X,w,func) :
        return func(np.dot(X,w.T))

    def forward(self,X,W,funcs) :
        for w,func in zip(W,funcs) :
            X = self.one_layer_out(X,w,func)
        return X

    # RNN forward function, 
    #          takes sequences as an (n x seq_len x value_len) np array 
    #          where value_len is the length of the vector of values
    # 
    #       returns all outputs and all hidden states as a tuple of np arrays
    #       where dimensions are Y: (seq_len x n x val_len)
    #                            H: (seq_len x n x hidden dims) 
    def RNN_forward(self,sequences,hid_func=tanh_func,out_func=sigmoid_func) :
        # this should work when seguences consist of vectors as well
        Y = []
        H = []
        hidden_state = np.zeros((sequences.shape[0],self._W.shape[0]))
        for t in range(sequences.shape[1]) :
            x = sequences[:,t]
            if len(x.shape) == 1  :
                x = np.array([x]).T
            # if hidden is deep NN make function to forward
                # Ux = self.forward(self._U@inp)
                # Wh = self.forward(self._W@hidden_state)
            Ux = np.dot(x,self._U)
            Wh = np.dot(hidden_state,self._W)
            h = hid_func(Ux + Wh)
            y = out_func(np.dot(h,self._V))
            H.append(h)
            Y.append(y)
        return (np.array(Y), np.array(H))

    def RNN_backward(self) :
        pass

    def train() :
        pass






if __name__ == "__main__" :


    rnn1D = RNN()
    rnn1D.init_weights((1,(100,),1))
    seq = np.array([[[1],[2],[3]],[[4],[5],[6]]])
    (Y,H) = rnn1D.RNN_forward(seq,tanh_func,sigmoid_func)
    print(H.shape)
    print(Y.shape)

    print("\n\n ---------------multi Dimensianal values\n\n")
    rnn3D = RNN()
    rnn3D.init_weights((4,(100,),4))
    seq3D = np.arange(2*3*4).reshape((2,3,4))
    (Y3,H3) = rnn3D.RNN_forward(seq3D,tanh_func,sigmoid_func)
    print(H3.shape)
    print(Y3.shape)


    # # Data
    # predict next number in a sine wave function
    sin = np.array([math.sin(x) for x in np.arange(200)])
    X = []
    Y = np.array([])

    seq_len = 50 
    N = len(sin) - seq_len

    for i in range(N-seq_len) :
        X.append(sin[i:i+seq_len])
        Y = np.append(Y,sin[i+seq_len])

    #X = np.array(X)
    X = expand_dims(X,axis=2)
    Y = expand_dims(Y,axis=1)

    #print(X.shape)
    #print(rnn.RNN_forward(X,tanh_func).shape)





    

