class experimentalIterator: # class construction

    from numpy import concatenate
    global concat
    concat=concatenate()
    from sklearn.preprocessing import LabelEncoder #BustamJos3 preprocess  #convert categorical to numerical
    global labelEncoder
    labelEncoder=LabelEncoder()
    from sklearn.impute import KNNImputer # nan imputation
    global kNNImputer
    kNNImputer=KNNImputer()
    from sklearn.preprocessing import OneHotEncoder #oneHotEncoding for categoric cols
    global oneHotEncoder
    oneHotEncoder=OneHotEncoder()
    from sklearn.preprocessing import StandardScaler # standardization
    global standardScaler
    standardScaler=StandardScaler()
    from sklearn.model_selection import train_test_split #BustamJos3 for models # import train_test split function
    from sklearn.model_selection import GridSearchCV # hyperparameter getter
    from sklearn.ensemble import RandomForestClassifier #import RandomForest
    import seaborn as sns #visualization
    import matplotlib.pyplot as plt
    

    def __init__(self, X, y):
        self._X = self.dtypeConvertion( X )#store as attribute X data of dataset
        self._y = self.dtypeConvertion( y )#same as above but y
        self._preprocessedX = None #for now
        self._preprocessedy = None #same as above
    
    def dtypeConvertion(self, df): #convert dtypes to best fit #previous to BustamJos3 preprocess
        return df.convert_dtypes(infer_objects=True)
    
    def typeSeeker(self, type): #seek for name of cols which match the type
        listDtype=[str(i) for i in (self._X.dtypes==type).loc[(self._X.dtypes==type)==True].index] # take name cols which are categorical
        return listDtype

    def convertToNumeric(self):
        listCategoric=self.typeSeeker('string')
        dfX=self._X[listCategoric].fillna('wanted').copy() #to handle oneHotEncoder, replace NaN values with 'wanted'
        for i in listCategoric: # labelEncoding and replacing consecutively
            array_to_replace=labelEncoder.fit_transform( dfX[i].values ).reshape(-1,1) #labelEncoding
            dfX[i]=array_to_replace
        return dfX

    def dropNotRelevant(self,listToDrop): # start of BustamJos3 preprocess
        self._X.drop(listToDrop, axis=1, inplace=True)
    
    def preProcess(self):
        XCategoric = self.convertToNumeric().values
        listNumeric=self.typeSeeker('Int64') #numeric col names
        XNumeric=self._X[listNumeric].fillna(-1).values # take the numeric types only
        # Transformation on categorical cols with labelEncoder and NaN Imputation categorical
        kNNImputer(n_neighbors=1, missing_values=len(labelEncoder.classes_)-1,weights='distance') #imputation for categoric
        XCategoric=kNNImputer.fit_transform( XCategoric ) #imputation
        oneHotEncoder(handle_unknown='error',sparse=False) # one
        convertedOHE=oneHotEncoder.fit_transform(XCategoric[:,[0,2]]) # tranformation to get convertion to OHE  #array with categories of n-categoric cols
        # .reshape on XCategoric for dimensionality  and concatenate
        convertedOHE= concat( (convertedOHE[:,:len(oneHotEncoder.categories_[0])-1], XCategoric[:,1].reshape((-1,1)), convertedOHE[:,len(oneHotEncoder.categories_[0])-1:2*len(oneHotEncoder.categories_[0])-1]), axis=1 )
        # nanImputation for numeric cols
        kNNImputer(n_neighbors=1, missing_values=-1,weights='distance')
        imputedNumeric=kNNImputer.fit_transform(XNumeric)
        imputedNumeric=standardScaler().fit_transform(imputedNumeric) # standardization
        self._preprocessedX=concat( (imputedNumeric, convertedOHE), axis=1 )
        #TODO
        #1. evaluate standardization only on numeric cols and then concantenate
        #2. make implementation of selected models according to discussed