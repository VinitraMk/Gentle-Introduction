from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class PolynomialFeatures:
    train=[]
    test = []
    columns = []
    poly_features = []
    poly_features_test = []
    poly_target = []
    poly_names = []

    def __init__(self,train,test):
        self.train = train
        self.test = test

    def get_polynomial_features(self,input_columns):
        self.columns = input_columns
        self.poly_features = self.train[self.columns+['TARGET']]
        self.poly_features_test = self.test[self.columns]

        imputer = SimpleImputer(strategy='median')
        self.poly_target = self.poly_features['TARGET']
        self.poly_features = self.poly_features.drop(columns=['TARGET'])
        self.poly_features = imputer.fit_transform(self.poly_features)
        self.poly_features_test = imputer.fit_transform(self.poly_features_test)
        poly_transformer = PolynomialFeatures(degree=3)
        poly_transformer.fit(self.poly_features)
        self.poly_features = poly_transformer.transform(self.poly_features)
        self.poly_features_test = poly_transformer.transform(self.poly_features_test)
        self.poly_names = poly_transformer.get_feature_names(input=self.columns)

        return self.poly_features, self.poly_features_test, self.poly_target, self.poly_names

    def append_data_polynom_features(self):
        self.poly_features = pd.DataFrame(self.poly_features,columns=poly_names)
        self.poly_features['TARGET'] = self.poly_target
        self.poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
        train_poly = train.merge(self.poly_features,on='SK_ID_CURR', how='left')
        self.poly_features_test = pd.DataFrame(self.poly_features_test,columns=poly_names)
        self.poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
        test_poly = test.merge(self.poly_features_test,on='SK_ID_CURR', how='left')
        train_poly, test_poly = train_poly.align(test_poly,join='inner',axis=1)

        return train_poly, test_poly

