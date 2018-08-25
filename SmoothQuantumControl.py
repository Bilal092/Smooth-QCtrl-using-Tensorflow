import numpy as np
import scipy as sp
import tensorflow as tf

#####################################################################
class Q_System:
    '''
    This class is used to define system object, which contains
    systems drift and initilizes control hamiltonians as zeros.

    Ho = Drift Hamiltonians
    Num_Ctrl = number of controls
    d = Dimension of quantum system

    '''
    def __init__ (self, Ho, Num_Ctrl, d, Num_har):
        self.Ho = Ho
        self.Num_Ctrl = Num_Ctrl
        self.d = d
        self.Controls = {}
        for i in range(1, self.Num_Ctrl + 1):
            self.Controls[i] = tf.zeros( [self.d, self.d], dtype = tf.complex64 )

    def drift(self):
        # returns drift
        return self.Ho


#######################################################################

def Sol_sys(Ho, Controls, Num_Ctrl, Num_har):
    '''
    This function integrates Schrodinger's equation using Adaptive Runge-Kutta Integrator built
    into Tensorflow.
    Ho = drift Hamiltonian
    Controls = Dictionary of Control hamiltonians
    Num_Ctrl = Number of Control Hamiltonians
    Num_harm = Number of Fourier harmonics

    '''
###########################################################################

def evolution(U, t, QS, Num_har, A, w):
    '''
    This function deals with dynamical evolution of quantum systems with time
    U = Vectorized unitary evolution operator of quantum systems
    QS = Quantum System class object
    A = matrix of harmonics, where A(j,k) represents j-th harmonic k-th control
    w = vector of fourier base frequencies
    '''
    f1  = 0
    f2  = 0
    f3  = 0

    dMdt = tf.zeros([QS.d**2 + QS.d**2 * QS.Num_Ctrl * Num_har + QS.d**2 * QS.Num_Ctrl , 1])
    for k = 1 : Num_har
        f1 = f1 + A[k,1] * sin( k * t * w[1] )
        f2 = f2 + A[k,2] * sin( k * t * w[2] )
        f3 = f3 + A[k,3] * sin( k * t * w[3] )

    H = QS.Ho +  f1  * QS.Controls[1] + f2  * QS.Controls[2] + f3  * QS.Controls[3]
    L = tf.contrib.kfac.utils.kronecker_product(tf.eye(QS.d), H )

    count = 17

#  This part related to derivatives with respect to waight matrices
    for l = 1 : QS.Num_Ctrl
         for k = 1 : Num_har
        dMdt(count:count + 15,1) = -1j * (  sin( k * t.* w(l)  ) * self.L{l} * M(1:16,1) + Lt*M(count:count + 15,1) )
        count = count + 16


# to be written latter


#########################################################################
if __name__ == '__main__':
    with tf.Session():
     sx = tf.constant([[0, 1],[1, 0]], dtype = tf.complex64)
     sy = tf.constant([[0, -1j],[1j, 0]], dtype = tf.complex64)
     sz = tf.constant([[1, 0],[0, -1]], dtype = tf.complex64)

     s = Q_System(sx, 2, 2)
     s.Controls[1] = sy
     s.Controls[2] = sz

     print(s.Controls[1])


