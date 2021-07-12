

'''
multiprocess
'''

from concurrent.futures import ProcessPoolExecutor,as_completed
from preprocess import load_data
from build_parameters import bpparams,rfparams,lrparaps
from build_models import BPModel,RFmodel,LRmodel

import time

def trainbp(p):
    m=BPModel(*p)
    x_train, y_train, x_test, y_test = load_data()
    s=m.train(x_train,y_train,x_test,y_test)
    return p,s

def trainrf(p):
    m=RFmodel(*p)
    x_train, y_train, x_test, y_test = load_data()
    s=m.train(x_train,y_train,x_test,y_test)
    return p,s

def trainlr(p):
    m=LRmodel(*p)
    x_train, y_train, x_test, y_test = load_data()
    s=m.train(x_train,y_train,x_test,y_test)
    return p,s

#Back Propagation model
if __name__ == "__main__":
    score_list = []
    with ProcessPoolExecutor(4) as executor:#How many processes
        tasks = [executor.submit(trainbp,p) for p in bpparams]#Fill in the model, and the corresponding parameters
        start=time.time()
        for f in as_completed(tasks):
            p,s=f.result()
            score_list.append([*p, s])
        end=time.time()
        takes = end - start
        print('BPmodel train takes: {:.2f}'.format(takes))

'''
#Rondem forest model

if __name__ == "__main__":
    score_list = []
    with ProcessPoolExecutor(2) as executor:#How many processes
        tasks = [executor.submit(trainrf,p) for p in rfparams]#Fill in the model, and the corresponding parameters
        start=time.time()
        for f in as_completed(tasks):
            p,s=f.result()
            score_list.append([*p, s])
        end=time.time()
        takes = end - start
        print('RFmodel train takes: {:.2f}'.format(takes))
'''

'''
#Logestic regression model

if __name__ == "__main__":
    score_list = []
    with ProcessPoolExecutor(2) as executor:#How many processes
        tasks = [executor.submit(trainlr,p) for p in lrparaps]#Fill in the model, and the corresponding parameters
        start=time.time()
        for f in as_completed(tasks):
            p,s=f.result()
            score_list.append([*p, s])
        end=time.time()
        takes = end - start
        print('LRmodel train takes: {:.2f}'.format(takes))
'''
