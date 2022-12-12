class experimentalIterator: # class construction

    from numpy import concatenate
    global concat
    concat=concatenate
    from sklearn.preprocessing import LabelEncoder #BustamJos3 preprocess  #convert categorical to numerical
    global labelEncoder
    labelEncoder=LabelEncoder()
    from sklearn.model_selection import train_test_split #BustamJos3 for models # import train_test split function
    global train_test
    train_test=train_test_split
    from sklearn.ensemble import RandomForestClassifier #import RandomForest
    global randomForestC
    randomForestC = RandomForestClassifier()
    import seaborn as sns #visualization
    import matplotlib.pyplot as plt
    
    def __init__(self, X, y):
        self._X = self.Prepocess_Spaceship_Titanic( X )#store as attribute X data of dataset
        self._y = labelEncoder.fit_transform(y.values) #to numeric
        self._preproX=0
        self._XTrain=0
        self._yTrain=0
        self._XTest=0
        self._yTest=0
    
    def Prepocess_Spaceship_Titanic(self, d):
        for i in range(len(d)):
            if (d['HomePlanet'].isna()[i] == True and type(d['Cabin'][i])!= float):
                if d['Cabin'][i][0] in 'ABCT':
                    d.loc[i,'HomePlanet']='Europa'
                elif d['Cabin'][i][0] == 'G':
                    d.loc[i,'HomePlanet'] = 'Earth'
            elif (d['RoomService'].isna()[i]== True and d['CryoSleep'][i]==True) or (d['RoomService'].isna()[i]== True and d['Age'][i]<=12):
                d.loc[i,'RoomService']= float(0)
            elif (d['FoodCourt'].isna()[i]== True and d['CryoSleep'][i]==True) or (d['FoodCourt'].isna()[i]== True and d['Age'][i]<=12):
                d.loc[i,'FoodCourt']= float(0)   
            elif (d['ShoppingMall'].isna()[i]== True and d['CryoSleep'][i]==True) or (d['ShoppingMall'].isna()[i]== True and d['Age'][0]<=12):
                d.loc[i,'ShoppingMall']= float(0) 
            elif (d['Spa'].isna()[i]== True and d['CryoSleep'][i]==True) or (d['Spa'].isna()[i]== True and d['Age'][i]<=12):
                d.loc[i,'Spa']= float(0) 
            elif (d['VRDeck'].isna()[i]== True and d['CryoSleep'][i]==True) or (d['VRDeck'].isna()[i]== True and d['Age'][i]<=12):
                d.loc[i,'VRDeck']= float(0)
            elif d['CryoSleep'].isna()[i]== True and d['RoomService'][i]==float(0) and d['FoodCourt'][i]==float(0) and d['ShoppingMall'][i]==float(0) and d['Spa'][i]==float(0) and d['VRDeck'][i]== float(0):
                d.loc[i,'CryoSleep']=True
            elif (d['CryoSleep'].isna()[i]== True and d['RoomService'][i] != 0  and d['FoodCourt'][i] != 0):
                d.loc[i,'CryoSleep'] = False
            elif (d['VIP'].isna()[i]== True and d['HomePlanet'][i]=='Earth'):
                d.loc[i,'VIP']=False
            elif (d['VIP'].isna()[i]== True and type(d['Cabin'][i])!= float):
                if d['Cabin'][i][0]=='G' or d['Cabin'][i][0]=='T':
                    d.loc[i,'VIP']=False
            elif (d['VIP'].isna()[i]== True and d['Age'][i]<25 and d['HomePlanet'][i]=='Europe'):
                d.loc[i,'VIP']=False
            elif (d['VIP'].isna()[i]== True and d['Age'][i]<18 and d['HomePlanet'][i]=='Mars'):
                d.loc[i,'VIP']=False
            elif (d['VIP'].isna()[i]== True and d['HomePlanet'][i] == 'Mars' and d['CryoSleep'][i]==True):
                d.loc[i,'VIP']=False
            elif (d['VIP'].isna()[i]== True and d['HomePlanet'][i] == 'Europa' and d['CryoSleep'][i] == True and d['Age'][i] < 25):
                d.loc[i,'VIP'] = False
            elif (d['Destination'].isna()[i]== True and d['Age'][i]>18 and d['RoomService'][i]== float(0) and d['FoodCourt'][i]== float(0) and d['ShoppingMall'][i]== float(0) and d['Spa'][i]== float(0) and d['VRDeck'][i]== float(0) and d['CryoSleep'][i]==False):
                d.loc[i,'Destination']= 'TRAPPIST-1e'
            elif i<(len(d)-4):
                for j in range(1,5): #Suponiendo una familia de max 5 integrantes
                    if type(d['Name'][i])!= float:
                        if type(d['Name'][i+j])!= float:
                            if d['Destination'].isna()[i]== True and (d['Name'][i].split()[1] == d['Name'][i+j].split()[1]):
                                d.loc[i,'Destination']=d['Destination'][i+j]
                    if d['HomePlanet'].isna()[i]== True and (d['PassengerId'][i][:4]==d['PassengerId'][i+j][:4]):
                        d.loc[i,'HomePlanet']=d['HomePlanet'][i+j]

            elif i>=(len(d)-4):
                for j in range(1,len(d)-i): 
                    if type(d['Name'][i])!= float:
                        if type(d['Name'][i+j])!= float:
                            if d['Destination'].isna()[i]== True and (d['Name'][i].split()[1] == d['Name'][i+j].split()[1]):
                                d.loc[i,'Destination']=d['Destination'][i+j]
                    if d['HomePlanet'].isna()[i]== True and (d['PassengerId'][i][:4]==d['PassengerId'][i+j][:4]):
                        d.loc[i,'HomePlanet']=d['HomePlanet'][i+j]
        return d
    
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
        self._preproX = concat( (imputedNumeric, convertedOHE), axis=1 )
        #TODO
        # 1. Import and tun Neural Network

    def trainTestSplit(self): #to implement models
        self._XTrain, self._yTrain, self._XTest, self._yTest=train_test(self._preproX, self.y, test_size=0.33, random_state=41)
    
    def neuralNetwork(self):
        import tensorflow as tf
        from sklearn.model_selection import GridSearchCV
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Dropout
        from scikeras.wrappers import KerasClassifier
        from tensorflow.keras.constraints import MaxNorm
        def create_model(neurons): # Function to create model, required for KerasClassifier
            model = Sequential() # create model
            model.add(Dense(neurons, input_shape=(14,), kernel_initializer='uniform', activation='linear', kernel_constraint=MaxNorm(4)))
            model.add(Dropout(0.2))
            model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        seed = 41 # fix random seed for reproducibility
        tf.random.set_seed(seed)
        model = KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=0) # create model
        return model
    
    def gridCVHyperparameter(self, model, params, metric):
        from sklearn.model_selection import GridSearchCV # hyperparameter getter
        gridSearchCV=GridSearchCV(model, params, scoring=metric)
        self.trainTestSplit()
        gridSearchCV.fit(self._XTrain, self._yTrain)
        return model()