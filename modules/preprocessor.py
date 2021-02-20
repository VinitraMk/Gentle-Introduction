import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        pass

    def missing_values_table(self,df):
        mis_val = df.isnull().sum()
        mis_val_percentage = 100 * mis_val / len(df)
        mis_val_table = pd.concat([mis_val,mis_val_percentage],axis=1)

        mis_val_table_ren_cols = mis_val_table.rename(columns={0:'Missing values',1:'% of Total Values'})

        mis_val_table_ren_cols = mis_val_table_ren_cols[mis_val_table_ren_cols.iloc[:,1]!=0].sort_values('% of Total Values',ascending=False).round(1)

        print('Your selected dataframe has '+str(df.shape[1])+' columns.\nThere are '+str(mis_val_table_ren_cols.shape[0])+' columns that have missing values\n\n')

        return mis_val_table_ren_cols

    def column_analysis(self,df):
        print('Number of each type of column')
        print(df.dtypes.value_counts(),'\n\n')
        print('Number of unique categories in each object column')
        print(df.select_dtypes('object').apply(pd.Series.nunique,axis=0),'\n\n')

    def encode_categorical_vars(self,df_train,df_test):
        le = LabelEncoder()

        for col in df_train:
            if(df_train[col].dtype=="object"):
                if(len(list(df_train[col].unique()))<=2):
                    le.fit(df_train[col])
                    df_train[col] = le.transform(df_train[col])
                    df_test[col] = le.transform(df_test[col])

        df_train = pd.get_dummies(df_train)
        df_test = pd.get_dummies(df_test)

        print('Training Features shape: ',df_train.shape)
        print('Testing Features shape: ',df_test.shape,'\n\n')

        return df_train,df_test

    def align_train_test(self,df_train,df_test):
        train_labels = df_train['TARGET']
        df_train,df_test = df_train.align(df_test,join='inner',axis=1)
        df_train['TARGET'] = train_labels

        print('Training Features shape: ',df_train.shape)
        print('Testing Features shape: ',df_test.shape,'\n\n')

        return df_train,df_test
