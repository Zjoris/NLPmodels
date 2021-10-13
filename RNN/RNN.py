'''
This file contains an implementation of a Recurrent Neural Network 
built from scratch using numpy, ... etc

This site was very helpful:
https://d2l.ai/chapter_recurrent-neural-networks/bptt.html



Notes to self or to do list: 
   - implement backpropagation
   - Implement deep hidden layer (weight initialization)
   - Add bias? 
   - Fix H,Y matrix in RNN_forward (seems a bit inefficient)
   - Add a link to an external file written in latex containing all 
     derivations.
'''
from sys import dont_write_bytecode
import numpy as np
import math

from numpy.lib.shape_base import expand_dims
from numpy.ma.core import concatenate


#### Activation Functions  ###

# A function to calculate the sigmoid of X (single value or array)
def sigmoid_func(X) :
    return 1/(1+np.exp(-X))

# A function to calculate the softmax of X (single value or array)
def softmax_func(X) :
    return np.exp(X)/np.sum(np.exp(X))

# A function to calculate the tanh of X (single value or array)
def tanh_func(X) :
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))


#### Loss functions ############

# Cross Entropy loss
def cross_entropy_loss(yhat, y) :
    return -y*np.log(yhat)


def total_loss(loss) :
    return sum(loss)

def average_loss(loss) :
    return sum(loss)/len(loss)



# Recurrent Neural Network
class RNN() :
    def __init__(self) :
        self._U = None
        self._W = None
        self._V = None
        self._D = None
        self._hidden_func = None
        self._out_func = None

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

    def recursiveU(self,dU,X,H,trunc) :
        if trunc == 0:
            return dU
        h0 = expand_dims(np.zeros((H.shape[1:])),axis=0)
        Hprev = np.concat((h0,H[:-1]))
        x0 = expand_dims(np.zeros((X.shape[1:])))
        Xprev = np.concatenate((x0,X[:-1]))
        if self._hidden_func == tanh_func :
            return (1-H**2) * (X + self._W * self.recursiveU(dU,Xprev,Hprev,trunc-1))

    def recursiveW(self,dW,H,trunc) :
        h0 = expand_dims(np.zeros((H.shape[1:])),axis=0)
        if trunc == 0 :
            return H
        if (H[-2] == h0).all() :
            return H
        # How the dimensions ?
        Hprev = np.concatenate((h0,H[:-1]))
        if self._hidden_func == tanh_func :
            print("\n",trunc)
            print(H)
            print(Hprev)
            #print(((1-H**2)*Hprev))
            print(H.shape)
            print(Hprev.shape)
            print(self._W.T.shape)
            print(np.matmul(self.recursiveW(dW,Hprev,trunc-1),self._W.T))
            return (1-H**2)* Hprev + (1-H**2) * np.matmul(self.recursiveW(dW,Hprev,trunc-1),self._W.T)

    # RNN forward function, 
    #          takes sequences as an (n x seq_len x value_len) np array 
    #          where value_len is the length of the vector of values
    # 
    #       returns all outputs and all hidden states as a tuple of np arrays
    #       where dimensions are Y: (seq_len x n x val_len)
    #                            H: (seq_len x n x hidden dims) 
    def RNN_forward(self,sequences,hid_func=tanh_func,out_func=sigmoid_func) :
        self._hidden_func = hid_func
        self._out_func = out_func
        X = []
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
            X.append(x)
        return (np.array(X),np.array(Y), np.array(H))


    # Maybe make general function that takes a function fx and dx that specifies 
    # what to partial on to calculate derivative for any function. 
    def calc_gradients(self,X,Y,Yhat,H,loss_func = "cross_entropy",trunc = 3) :
        dU = np.zeros(self._U.shape)
        dW = np.zeros(self._W.shape) # grotere list, reken voor elke gradient op, sum voor trunc steps en sum al die dingen bij elkaar op
        dV = np.zeros(self._V.shape)
        if loss_func == "cross_entropy" and self._out_func == sigmoid_func :
            # find gradients for each step in time
            dEdHV = -Y * (1-Yhat)
            print("dE/dHV shape: ",dEdHV.shape)
            print(self._V.T.shape)
            dEdH = np.matmul(dEdHV,self._V.T)
            print("dE/dH: shape: ",dEdH.shape)
            HT = H.transpose((0,2,1))
            # matrix multiplication to calculate dV for each time step, sum up to get dV
            dV = sum(np.matmul(HT,dEdHV))
            #dU = sum(np.matmul(self.recursiveU(dU,X,H,trunc),dEdH))
            dW = sum(np.matmul(self.recursiveW(dW,H,trunc).transpose((0,2,1)),dEdH))
            return (dU, dW, dV)

            for t in range(Y.shape[0])[::-1] :
                pass
            


            for (y,yhat,h) in zip(Y,Yhat,H) :
                # dV
                # This is not necessary anymore for it is done using 1 matrix multiplication
                dedhV = -y *(1-yhat)
                #dV += np.dot(h.T,dedhV) # ? d ht*V/d V = ht ? V - alpha dV wouldn't work because dV has different dimension from V 
                # Okay so when n=1 and the values provided to the network are scalars
                # we get that dY/dV = (Y[1x1] -Y[1x1]^2) * h.T[hidden_dimx1] == V.shape This is good because we end up with V shape
                # Which could also be h.T * Y(1 - Y)
                # 
                # now when the values are actually vectors we get 
                # dY/dV = Y[1x3] * (1-Y[1x3]) * a matrix that is hidden dims x vec_len x vec_len
                # let's call Y * (1-Y) f(Y)
                # And because 1x3 * 3*1 = 1x1 we have to dot product Y *(1-Y) with each h vector in dhV/dV which results in
                #   [ f(Y)11*h11, f(Y)12*h11 ...  f(Y)1vec_len *h11 ] 
                #   [ f(Y)11*h12, ....]
                #   [ f(Y)11*h13,     ]
                #   [ ....]
                #   [ f(Y)11*h1hiddims, .... ]
                #   = h.T * f(Y) = h.T * Y*(1-Y)
                #
                # Now let's say N>1
                #   = h.T * Y*(1-Y)


                #dU
                #print(self._V.shape)
                #print(dedhV.shape)
                #print(np.dot(dedhV,self._V.T).shape)
                dedh = np.matmul(dedhV,self._V.T)
                print(dedhV in dEdHV)
                print(dedh in dEdH)
                #print(dedh.shape)
                #print((1-h**2))


                
            

            #dW
            #print(dV.shape)
            #print(dV1 == dV)
            return (dU, dW, dV)

    def update_weight(self,learning_rate) :
        (dU, dW, dV) = self.calc_gradients()
        self._U = self._U - learning_rate * dU
        self._W = self._W - learning_rate * dW
        self._U = self._V - learning_rate * dV
        pass


    def RNN_backward(self) :
        pass

    def train() :
        # in loop 
        #   forward pass
        #   loss
        #   backward pass
        pass






if __name__ == "__main__" :


    rnn1D = RNN()
    rnn1D.init_weights((1,(3,),1))
    rnn1D._out_func = sigmoid_func
    seq = np.array([[[1],[2],[3]],[[6],[7],[8]]])
    (X,Y,H) = rnn1D.RNN_forward(seq,tanh_func,sigmoid_func)
    print(H.shape)
    print(Y.shape)
    (dU,dW,dV) = (rnn1D.calc_gradients(X,Y,Y,H,"cross_entropy"))

    print("\n\n ---------------multi Dimensianal values\n\n")
    rnn3D = RNN()
    rnn3D.init_weights((4,(100,),4))
    rnn3D._out_func = sigmoid_func
    seq3D = np.arange(2*3*4).reshape((2,3,4))
    (X3,Y3,H3) = rnn3D.RNN_forward(seq3D,tanh_func,sigmoid_func)
    print("H: shape (t, n, hidden_dims): ",H3.shape)
    print("Y: shape (t, n, val_len): ",Y3.shape)
    print("X: shape (t, n, val_len): ", X3.shape)
    (dU3,dW3,dV3) = (rnn3D.calc_gradients(X3,Y3,Y3,H3,"cross_entropy"))
    #print(-(Y3*np.log(Y3)))
    #print(sum(cross_entropy_loss(Y3,Y3)))


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





    

