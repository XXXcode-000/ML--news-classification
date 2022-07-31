import numpy as np


class CPerceptron( object ):
    def __init__(self, train_samples, train_Y, category, time=40000, alpha=0.1):
        self.X = train_samples
        self.label = train_Y
        self.Y = []
        self.W = []
        self.b = None
        self.alpha = alpha
        self.time = time
        self.category = category
        self.study()

    def study(self):
        for i in range( len( self.label ) ):
            self.Y.append( 1 if self.label[i] == self.category else -1 )
        self.W = np.full( shape=np.shape( self.X[0] ), fill_value=0.0 )
        self.b = 0
        t = 0
        i = 0
        while i < np.shape( self.X )[0]:
            loss = self.Y[i] * (np.dot( self.W, self.X[i] ) + self.b)
            if loss <= 0:
                self.W = self.W + self.alpha * self.X[i] * self.Y[i]
                self.b = self.b + self.alpha * self.Y[i]
                # print( "loss=%.2f,分类错误," % loss, "更新为W:", self.W, "b:", self.b )
                i = 0
            else:
                i += 1
            t += 1
            if t >= self.time:
                break
        return self.W, self.b


def perceptron(train_x, train_y, test_x, test_y, category, time=40000, alpha=0.1):
    acc = 0
    w = []
    b = []
    for i in range( category ):
        pi = CPerceptron( train_x, train_y, i, alpha=alpha, time=time)
        w.append( pi.W )
        b.append( pi.b )
    for i in range( len( test_x ) ):
        loss = [np.dot( w[j], test_x[i] ) + b[j] for j in range( category )]
        predict = loss.index( max( loss ) )
        if predict == test_y[i]:
            acc += 1
    return acc / len( test_x )
