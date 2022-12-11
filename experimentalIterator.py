class experimentalIterator: # class construction

    from numpy import concatenate
    global concat
    concat=concatenate
    from sklearn.preprocessing import LabelEncoder #BustamJos3 preprocess  #convert categorical to numerical
    global labelEncoder
    labelEncoder=LabelEncoder()
    from sklearn.model_selection import train_test_split #BustamJos3 for models # import train_test split function
    from sklearn.model_selection import GridSearchCV # hyperparameter getter
    from sklearn.ensemble import RandomForestClassifier #import RandomForest
    import seaborn as sns #visualization
    import matplotlib.pyplot as plt
    

    def __init__(self, X, y):
        self._X = X #store as attribute X data of dataset
        self._y = y #same as above but y
    
    def dtypeConvertion(self, dataFrame): #convert dtypes to best fit #previous to BustamJos3 preprocess
        df=dataFrame.copy()
        return df.convert_dtypes(infer_objects=True)
    
    def typeSeeker(self, type, dataFrame): #seek for name of cols which match the type
        df=self.dtypeConvertion(dataFrame).copy()
        listDtype=[str(i) for i in (df.dtypes==type).loc[(df.dtypes==type)==True].index] # take name cols which are categorical
        return listDtype

    def convertToNumeric(self, dataFrame):
        df=dataFrame.copy()
        df.drop(['PassengerId','Name'], axis=1,inplace=True)
        listCategoric=self.typeSeeker('string',df)
        df.loc[:,listCategoric].fillna('wanted',inplace=True) #to handle oneHotEncoder, replace NaN values with 'wanted'
        for i in listCategoric: # labelEncoding and replacing consecutively
            array_to_replace=labelEncoder.fit_transform( df.loc[:,i].values ) #labelEncoding
            to_search=len(labelEncoder.classes_)-1 #store No. of categorie for 'wanted
            for j in range(len(array_to_replace)):
                if array_to_replace[j] == to_search:
                    array_to_replace[j]=-1
            df.loc[:,i]=array_to_replace
        return df.loc[:,listCategoric].values

    def preProcess(self):  # start of BustamJos3 preprocess
        XCategoric = self.convertToNumeric(self._X)
        # Transformation on categorical cols with labelEncoder and NaN Imputation categorical
        from sklearn.impute import KNNImputer # nan imputation
        KNNImputer(n_neighbors=1, missing_values=-1,weights='distance', copy=False) #imputation for categoric
        KNNImputer().fit_transform( XCategoric ) #imputation
        from sklearn.preprocessing import OneHotEncoder #oneHotEncoding for categoric cols
        oneHotEncoder=OneHotEncoder(handle_unknown='error', sparse=False)
        convertedOHE=oneHotEncoder.fit_transform(XCategoric[:,[0,2]]) # tranformation to get convertion to OHE  #array with categories of n-categoric cols
        # .reshape on XCategoric for dimensionality  and concatenate
        convertedOHE= concat( (convertedOHE.reshape(-1,8), XCategoric[:,1].reshape(-1,1)), axis=1 )
        # nanImputation for numeric cols
        listNumeric=self.typeSeeker('Int64', self._X) #numeric col names
        XNumeric=self._X[listNumeric].fillna(-1).values # take the numeric types only
        KNNImputer(n_neighbors=1, missing_values=-1,weights='distance')
        imputedNumeric=KNNImputer().fit_transform(XNumeric)
        from sklearn.preprocessing import StandardScaler # standardization
        imputedNumeric=StandardScaler().fit_transform(imputedNumeric) # standardization
        return concat( (imputedNumeric, convertedOHE), axis=1 )
        #TODO
        #1. evaluate standardization only on numeric cols and then concantenate
        #2. make implementation of selected models according to discussed