import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from itertools import zip_longest

class bankmarketing:

    bank = ''
    accuracy = 0
    bank_all = dict()

    def __init__(self):  
         self.bank = pd.read_csv(r'C:\Users\Gintoki\source\repos\FlaskWebProject1\FlaskWebProject1\FlaskWebProject1\bank-additional-full.csv', sep = ';')
         
    def dataframe(self):
        return self.bank

    def encode(self):
        y = pd.get_dummies(self.bank['y'], columns = ['y'], prefix = ['y'], drop_first = True)
        bank_client = self.bank.iloc[: , 0:7]
        labelencoder_X = LabelEncoder()
        bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
        bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
        bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
        bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
        bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
        bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 

        def age(dataframe):
            dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
            dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
            dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
            dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
            return dataframe

        age(bank_client);

        bank_related = self.bank.iloc[: , 7:11]
        self.bank[(self.bank['duration'] == 0)]

        labelencoder_X = LabelEncoder()
        bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
        bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
        bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 

        def duration(data):

            data.loc[data['duration'] <= 102, 'duration'] = 1
            data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
            data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
            data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
            data.loc[data['duration']  > 644.5, 'duration'] = 5

            return data
        duration(bank_related);

        bank_se = self.bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
        bank_o = self.bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
        bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)

        bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
        bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                             'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                             'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
        
        #self.bank_all = dict(zip_longest(['bank_client', 'bank_related', 'bank_se, bank_o'], [bank_client, bank_related, bank_se, bank_o]))
        from itertools import chain, repeat
        self.bank_all = dict(zip(['bank_client', 'bank_related', 'bank_se', 'bank_o'], chain([bank_client, bank_related, bank_se, bank_o], repeat(None))))
        return bank_final, y
       

    def preprocessing(self):
        bank_final, y = self.encode()

        X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.1942313295, random_state = 101)

        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        return X_train, X_test, y_train, y_test


    def logregression(self):
        X_train, X_test, y_train, y_test = self.preprocessing()
        logmodel = LogisticRegression() 
        logmodel.fit(X_train,y_train)
        logpred = logmodel.predict(X_test)

        
        self.accuracy = round(accuracy_score(y_test, logpred),2)*100
        #LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
        return self.accuracy