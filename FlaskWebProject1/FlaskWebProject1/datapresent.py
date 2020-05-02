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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_linear_regression

from itertools import zip_longest
from itertools import chain, repeat

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

class bankmarketing:

    bank = ''
    bank_all = dict()
    data = ''
    bank_kmeans = ''
    cluster_centers = np.array([[-1.5,-1.5], [-1.6,-0.5], [-1.7,0.5], [-1.9,1.5], [-2,2.5],
                                [0.5,-1], [0,0], [0,1], [-0.2,2], [-0.5, 2.8],
                                [3,-1], [3,0], [2.5,1.1], [2.5,2], [2.5,3.2]])

    def __init__(self):  
         self.bank = pd.read_csv(r'C:\Users\Gintoki\source\repos\FlaskWebProject1\FlaskWebProject1\FlaskWebProject1\bank-additional-full.csv', sep = ';')
         self.data = self.bank.iloc[: , 0:-1]
         self.bank_kmeans = self.bank.iloc[: , 0:-1]
         
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



    def plot_age_1(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        sns.countplot(x = 'age', data = self.data)
        ax.set_xlabel('Age', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_title('Age Count Distribution', fontsize=15)
        sns.despine()
        return fig

    def plot_age_2(self):
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
        sns.boxplot(x = 'age', data = self.data, orient = 'v', ax = ax1)
        ax1.set_xlabel('People Age', fontsize=15)
        ax1.set_ylabel('Age', fontsize=15)
        ax1.set_title('Age Distribution', fontsize=15)
        ax1.tick_params(labelsize=15)

        sns.distplot(self.data['age'], ax = ax2)
        sns.despine(ax = ax2)
        ax2.set_xlabel('Age', fontsize=15)
        ax2.set_title('Age Distribution', fontsize=15)
        ax2.tick_params(labelsize=15)

        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout() 
        return fig

    def plot_jobs(self):
     
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)
        sns.countplot(x = 'job', data = self.data)
        ax.set_xlabel('Job', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_title('Age Count Distribution', fontsize=15)
        ax.tick_params(labelsize=15)
        sns.despine()
        return fig

    def plot_marital(self):
        
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        sns.countplot(x = 'marital', data = self.data)
        ax.set_xlabel('Marital', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_title('Age Count Distribution', fontsize=15)
        ax.tick_params(labelsize=15)
        sns.despine()
        return fig

    def plot_education(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 5)
        sns.countplot(x = 'education', data = self.data)
        ax.set_xlabel('Education', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_title('Education Count Distribution', fontsize=15)
        ax.tick_params(labelsize=15)
        sns.despine()
        return fig

    def plot_DHL(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
        sns.countplot(x = 'default', data = self.data, ax = ax1, order = ['no', 'unknown', 'yes'])
        ax1.set_title('Default', fontsize=15)
        ax1.set_xlabel('')
        ax1.set_ylabel('Count', fontsize=15)
        ax1.tick_params(labelsize=15)

        sns.countplot(x = 'housing', data = self.data, ax = ax2, order = ['no', 'unknown', 'yes'])
        ax2.set_title('Housing', fontsize=15)
        ax2.set_xlabel('')
        ax2.set_ylabel('Count', fontsize=15)
        ax2.tick_params(labelsize=15)

        sns.countplot(x = 'loan', data = self.data, ax = ax3, order = ['no', 'unknown', 'yes'])
        ax3.set_title('Loan', fontsize=15)
        ax3.set_xlabel('')
        ax3.set_ylabel('Count', fontsize=15)
        ax3.tick_params(labelsize=15)

        plt.subplots_adjust(wspace=0.25)
        return fig

    def plot_duration(self):
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
        sns.boxplot(x = 'duration', data = self.data, orient = 'v', ax = ax1)
        ax1.set_xlabel('Calls', fontsize=10)
        ax1.set_ylabel('Duration', fontsize=10)
        ax1.set_title('Calls Distribution', fontsize=10)
        ax1.tick_params(labelsize=10)

        sns.distplot(self.data['duration'], ax = ax2)
        sns.despine(ax = ax2)
        ax2.set_xlabel('Duration Calls', fontsize=10)
        ax2.set_title('Duration Distribution', fontsize=10)
        ax2.tick_params(labelsize=10)

        plt.subplots_adjust(wspace=0.5)
        plt.tight_layout() 

        return fig

    def plot_con_day_month(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
        sns.countplot(self.data['contact'], ax = ax1)
        ax1.set_xlabel('Contact', fontsize = 10)
        ax1.set_ylabel('Count', fontsize = 10)
        ax1.set_title('Contact Counts')
        ax1.tick_params(labelsize=10)

        sns.countplot(self.data['month'], ax = ax2, order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        ax2.set_xlabel('Months', fontsize = 10)
        ax2.set_ylabel('')
        ax2.set_title('Months Counts')
        ax2.tick_params(labelsize=10)

        sns.countplot(self.data['day_of_week'], ax = ax3)
        ax3.set_xlabel('Day of Week', fontsize = 10)
        ax3.set_ylabel('')
        ax3.set_title('Day of Week Counts')
        ax3.tick_params(labelsize=10)

        plt.subplots_adjust(wspace=0.25)

        return fig

    def linear(self):
        X_train, X_test, y_train, y_test = self.preprocessing()

        linreg = LinearRegression(normalize=True) 
        linreg.fit(X_train,y_train)
        linpred = linreg.predict(X_test)


        r2 = round(metrics.r2_score(y_test,linpred), 2)
        mae = round(metrics.mean_absolute_error(y_test,linpred),2)
        mse = round(metrics.mean_squared_error(y_test,linpred),2)
        coef = linreg.coef_
        return r2, mae, mse, coef, linpred
        

    def plot_linear(self):
        bank_final, y = self.encode()
        bank_final = bank_final[:, np.newaxis, 2]
        X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.5, random_state = 101)

        r2, mae, mse, coef, linpred = self.linear()
        fig, ax = plt.subplots()
        
        ax.scatter(X_test, y_test,  color='black')
        ax.plot(X_test, linpred, color='blue', linewidth=3)

        ax.xticks(())
        ax.yticks(())
        return fig


    def kmeans(self):
        self.bank_kmeans['campaign'] = self.bank_kmeans['campaign'].apply(lambda x: x if x<5 else 5) 
        def treat_pdays(value):
            if value <= 10:
                return 1
            if value > 10 and value <= 27:
                return 2
            if value > 27:
                return 0

        self.bank_kmeans['pdays'] = self.bank_kmeans['pdays'].apply(treat_pdays)

        def treat_previous(value):
    
            if value == 0:
                return 0
            if value == 1:
                return 1
            else:
                return 2
    
        self.bank_kmeans['previous'] = self.bank_kmeans['previous'].apply(treat_previous)


        catg_cols = ['job', 'marital', 'education', 'default', 'housing',
             'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
             'poutcome']

        num_cols = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                    'euribor3m', 'nr.employed']

        dummy_features = pd.get_dummies(self.bank_kmeans[catg_cols])

        num_features = self.bank_kmeans[num_cols]

        scaler = StandardScaler()

        num_features = pd.DataFrame(scaler.fit_transform(num_features), columns=num_features.columns)

        preprocessed_df = pd.concat([dummy_features, num_features], axis=1)

        labels = self.bank['y'].map({'no':0, 'yes':1})

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(preprocessed_df)

        pcs_df = pd.DataFrame(pcs)

        n_clusters = 15 

        kmeans = KMeans(n_clusters=15, max_iter=10000, verbose=1, init=self.cluster_centers)

        clusters = kmeans.fit_predict(pcs_df)

        pcs_df['cluster'] = clusters

        return pcs, pcs_df, labels

    def plot_kmeans_1(self):
        pcs, pcs_df, labels = self.kmeans()
        def plot_2d(X, y):   
            fig, ax = plt.subplots()
            for l in zip(np.unique(y)):
                ax.scatter(X[y==l, 0], X[y==l, 1], label=l)
            ax.legend(loc='upper right')
            return fig
        
        return plot_2d(pcs, labels)


    def plot_kmeans_2(self):
        pcs, pcs_df, labels = self.kmeans()
        fig, ax = plt.subplots()
        ax.scatter(pcs_df[0], pcs_df[1], c=pcs_df['cluster'])
        
        return fig

    def knn(self):
        X_train, X_test, y_train, y_test = self.preprocessing()

        knn = KNeighborsClassifier(n_neighbors=22)
        knn.fit(X_train, y_train)
        knnpred = knn.predict(X_test)

        conf = confusion_matrix(y_test, knnpred)
        accuracy = accuracy_score(y_test, knnpred)*100

        return conf, accuracy

    def plot_knn_conf(self):
        conf, accuracy = self.knn()
        fig, ax = plt.subplots()
        sns.heatmap(conf, annot=True, ax=ax)
        return fig


    def xgboost(self):
        bank_final, y = self.encode()
        X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.1942313295, random_state = 101)

        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        xgbprd = xgb.predict(X_test)

        conf = confusion_matrix(y_test, xgbprd )
        accuracy = accuracy_score(y_test, xgbprd)*100

        return conf, accuracy

    def plot_xgboost_conf(self):
        conf, accuracy = self.xgboost()
        fig, ax = plt.subplots()
        sns.heatmap(conf, annot=True, ax=ax)
        return fig