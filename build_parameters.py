
'''
Building Parameter Space
'''


lr_list=[0.01,0.03,0.001,0.003,0.0001,0.0003]
layer_list=[5,10,15,20]
units_list=[100,200,300,400]
batch_size_list=[64,128,256,512]
epochs_list=[5,10]

bpparams=[]
for lr in lr_list:
    for layer in layer_list:
        for units in units_list:
            for batch_size in batch_size_list:
                for epochs in epochs_list:
                    bpparams.append([lr,layer,units,batch_size,epochs])

n_estimators=[60,70,80,90,100]
max_depth=[2,4,5,7,8,10]
min_samples_leaf=[2,4,5,7,8,10]
max_features=[2,4,6]

rfparams=[]
for est in n_estimators:
    for depth in max_depth:
        for leaf in min_samples_leaf:
            for features in max_features:
                rfparams.append([est,depth,leaf,features])

C=[0.1,0.2,0.3,0.01,0.02,0.03,0.001,0.002,0.003]

lrparaps=[]
for c in C:
    lrparaps.append([c])

if __name__=='__main__':
    pass