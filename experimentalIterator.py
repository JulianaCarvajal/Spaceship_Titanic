class experimentalIterator: # class construction
    import pandas as pd #libraries


    from sklearn.model_selection import train_test_split #BustamJos3 for models # import train_test split function
    from sklearn.model_selection import GridSearchCV # hyperparameter getter
    from sklearn.ensemble import RandomForestClassifier #import RandomForest
    import seaborn as sns #visualization
    import matplotlib.pyplot as plt
    

    def __init__(self, X, y):
        self._X = self.dtypeConvertion( X )#store as attribute X data of dataset
        self._y = self.dtypeConvertion( y )#same as above but y
        self._X = self.dtypeConvertion( X )#for now, same as self._X
        self._y = self.dtypeConvertion( y )#same as above
    
    def dtypeConvertion(self, df): #convert dtypes to best fit #previous to BustamJos3 preprocess
        return df.convert_dtypes(infer_objects=True)
    
    def dropNotRelevant(self,listToDrop): # start of BustamJos3 preprocess
        self._X.drop(listToDrop, axis=1, inplace=True)
    
    def preProcess(self):
        listCategoric=[str(i) for i in (self._X.dtypes=='string').loc[(self._X.dtypes=='string')==True].index] # take name cols which are categorical
        XCategoric=self._X[listCategoric].fillna('wanted').values #to handle oneHotEncoder, replace NaN values with 'wanted'
        listNumeric=[str(i) for i in (self._X.dtypes=='Int64').loc[(self._X.dtypes=='Int64')==True].index] #numeric col names
        XNumeric=self._X[listNumeric].fillna(-1).values # take the numeric types only
        # Transformation on categorical cols with labelEncoder and NaN Imputation categorical
        from sklearn.preprocessing import LabelEncoder #BustamJos3 preprocess  #convert categorical to numerical
        from sklearn.impute import KNNImputer # nan imputation
        labelEncoder=LabelEncoder() #instancing
        dict_to_replace={}
        for i in range(XCategoric.shape[1]): # labelEncoding and replacing consecutively
            array_to_replace=labelEncoder.fit_transform( XCategoric[:,i] ).reshape(-1,1) #labelEncoding
            kNNImputer=KNNImputer(n_neighbors=1, missing_values=len(labelEncoder.classes_)-1,weights='distance') #imputation for categoric
            array_to_replace=kNNImputer.fit_transform( array_to_replace ) #imputation
            dict_to_replace[i]=array_to_replace
        import numpy as np # to concatenation
        XCategoric=np.concatenate( (dict_to_replace[0], dict_to_replace[1], dict_to_replace[2]), axis=1 ) #concatenate
        from sklearn.preprocessing import OneHotEncoder #oneHotEncoding for categoric cols
        oneHotEncoder=OneHotEncoder(handle_unknown='error',sparse=False) # instancing
        convertedOHE=oneHotEncoder.fit_transform(XCategoric[:,[0,2]]) # tranformation to get convertion to OHE  #array with categories of n-categoric cols
        # .reshape on XCategoric for dimensionality  and concatenate
        convertedOHE= np.concatenate( (convertedOHE[:,:len(oneHotEncoder.categories_[0])-1], XCategoric[:,1].reshape((-1,1)), convertedOHE[:,len(oneHotEncoder.categories_[0])-1:7 ]), axis=1 )
        # nanImputation for numeric cols
        kNNImputer=KNNImputer(n_neighbors=1, missing_values=-1,weights='distance')
        imputedNumeric=kNNImputer.fit_transform(XNumeric)
        from sklearn.preprocessing import StandardScaler # standardization
        imputedNumeric=StandardScaler().fit_transform(imputedNumeric) # standardization
        self._X=np.concatenate( (imputedNumeric, convertedOHE), axis=1 )
        #TODO
        #1. evaluate standardization only on numeric cols and then concantenate
        #2. make implementation of selected models according to discussed