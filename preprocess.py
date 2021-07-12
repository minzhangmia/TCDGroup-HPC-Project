
'''
Read the data
'''

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    data.pop('nameOrig')
    data.pop('nameDest')
    oe = OrdinalEncoder().fit(data.loc[:, ['type']])
    data.type = oe.transform(data.loc[:, ['type']])

    y = data.pop('isFraud')
    x = data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



    return x_train[:1000], y_train[:1000], x_test[:1000], y_test[:1000]

if __name__=='__main__':
    pass