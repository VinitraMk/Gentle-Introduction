
class DomainFeatures:
    train = []
    test = []

    def __init__(self,train,test):
        self.train = train
        self.test = test

    def get_domain_features(self):
        self.train['CREDIT_INCOME_PERCENT'] = self.train['AMT_CREDIT'] / self.train['AMT_INCOME_TOTAL']
        self.train['ANNUITY_INCOME_PERCENT'] = self.train['AMT_ANNUITY'] / self.train['AMT_INCOME_TOTAL']
        self.train['CREDIT_TERM'] = self.train['AMT_ANNUITY'] / self.train['AMT_CREDIT']
        self.train['DAYS_EMPLOYED_PERCENT'] = self.train['DAYS_EMPLOYED'] / self.train['DAYS_BIRTH']
        self.test['CREDIT_INCOME_PERCENT'] = self.test['AMT_CREDIT'] / self.test['AMT_INCOME_TOTAL']
        self.test['ANNUITY_INCOME_PERCENT'] = self.test['AMT_ANNUITY'] / self.test['AMT_INCOME_TOTAL']
        self.test['CREDIT_TERM'] = self.test['AMT_ANNUITY'] / self.test['AMT_CREDIT']
        self.test['DAYS_EMPLOYED_PERCENT'] = self.test['DAYS_EMPLOYED'] / self.test['DAYS_BIRTH']

        return self.train,self.test

