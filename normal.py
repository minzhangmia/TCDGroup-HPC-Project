
'''
Single-process sequential execution
'''

import time
from preprocess import load_data
from build_parameters import bpparams,rfparams,lrparaps
from build_models import BPModel,RFmodel,LRmodel


def train(model,params_list):
    score_list = []
    for p in params_list:
        clf=model(*p)
        x_train, y_train, x_test, y_test = load_data()
        s=clf.train(x_train, y_train, x_test, y_test)
    score_list.append([*p, s])
    return score_list

#Back Propagation model

start=time.time()
score_list=train(BPmodel,bpparams[:20])#Fill in the model and corresponding parameters
end=time.time()

takes=end-start
print('BPmodel train takes: {:.2f}'.format(takes))


#Rondem forest model
'''
start=time.time()
score_list=train(RFmodel,rfparams)#Fill in the model and corresponding parameters
end=time.time()

takes=end-start
print('RFmodel train takes: {:.2f}'.format(takes))
'''


#Linear regression model
'''
start=time.time()
score_list=train(LRmodel,LRparams)#Fill in the model and corresponding parameters
end=time.time()

takes=end-start
print('LRmodel train takes: {:.2f}'.format(takes))
'''
