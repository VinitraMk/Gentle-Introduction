import matplotlib.pyplot as plt
import numpy as np

class Cleanup:

    def remove_anomalies(self,df_train,df_test):


        #df_train['DAYS_EMPLOYED'].plot.hist(title='Days Employed Histogram')
        #plt.show()
        anom = df_train[df_train['DAYS_EMPLOYED']==365243]
        non_anom = df_train[df_train['DAYS_EMPLOYED']!=365243]
        print('The non-anomalies default on %0.2f%% of loans'%(100 * non_anom['TARGET'].mean()))
        print('The anomalies default on %0.2f%% of loans'%(100 * anom['TARGET'].mean()))
        print('There are %d anomolous days of employment'%(len(anom)),'\n\n')


        df_train['DAYS_EMPLOYED_ANOM'] = df_train['DAYS_EMPLOYED']==365243
        df_test['DAYS_EMPLOYED_ANOM'] = df_test['DAYS_EMPLOYED']==365243
        df_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
        df_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
        #df_train['DAYS_EMPLOYED'].plot.hist(title="Days Employment Histogram")
        #plt.show()

        return df_train,df_test
