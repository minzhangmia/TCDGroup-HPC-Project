
'''
Build models
'''


import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class BPModel:
    def __init__(self,lr,layers,units,batch_size,epochs):#Model initialization parameters
        self.seq=tf.keras.Sequential()
        self.lr=lr
        self.layers=layers
        self.units=units
        self.batch_size=batch_size
        self.epochs=epochs
        for i in range(layers):
            if i ==0:
                self.seq.add(tf.keras.layers.Dense(units=units,activation='relu'))
                units=int(units/2)
        self.seq.add(tf.keras.layers.Dense(units=2))

        self.seq.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics='acc')


    def train(self,x_train,y_train,x_test,y_test):
        self.seq.fit(x_train,y_train,epochs=self.epochs,verbose=0,
                     validation_data=(x_test,y_test),workers=4,batch_size=self.batch_size)
        score=self.seq.evaluate(x_test,y_test)[1]
        print('*'*100)
        return score

class RFmodel:
    def __init__(self,est,depth,leaf,feature):#Model initialization parameters
        self.est=est
        self.depth=depth
        self.leaf=leaf
        self.feature=feature
        self.clf=RandomForestClassifier(n_estimators=est,max_depth=depth,min_samples_leaf=leaf,max_features=feature)

    def train(self,x_train,y_train,x_test,y_test):
        self.clf.fit(x_train,y_train)
        return self.clf.score(x_test,y_test)

class LRmodel:
    def __init__(self,c):#Model initialization parameters
        self.c=c
        self.clf=LogisticRegression(C=c)
    def train(self,x_train,y_train,x_test,y_test):
        self.clf.fit(x_train,y_train)
        return self.clf.score(x_test,y_test)

if __name__=='__main__':
    pass