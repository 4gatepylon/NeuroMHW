# NOTE this is what they did for the students dataset

# Some nonsense to help you select features that will best predict the label
# y=pd.get_dummies(df['user-definedlabeln'])
# mi_score=mutual_info_classif(df.drop('user-definedlabeln',axis=1),df['user-definedlabeln'])
# mi_score=pd.Series(mi_score,index=df.drop('user-definedlabeln',axis=1).columns)
# mi_score=(mi_score*100).sort_values(ascending=False)
# print(mi_score)

# Selects the top 14 features
# print(mi_score.head(14).index)
# top_fea=['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
#    'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']


# Set to zero mean and unit variance (i.e. divide by variance). This assumes thin tails.
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# df_sc=StandardScaler().fit_transform(df[top_fea])

# TODO pytorch this shit
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import callbacks,layers

# TODO train/test split
# from sklearn.model_selection import train_test_split
# Xtr,xte,Ytr,yte=train_test_split(df_sc,y,random_state=108,test_size=0.27)
# xtr,xval,ytr,yval=train_test_split(Xtr,Ytr,random_state=108,test_size=0.27)

# TODO this is their model, probably too big for what we want to run, but I could be wrong!
# I'm willing to bet their network is overfitted
# Model-Building step, stacking the hidden layers
# model=keras.Sequential([
#     layers.Dense(64,input_shape=(14,),activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.27),
#     layers.Dense(124,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(248,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.32),   
#     layers.Dense(512,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.27),   
#     layers.Dense(664,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(512,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.32),
#     layers.Dense(264,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.27),
#     layers.Dense(124,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(2,activation='sigmoid')
# ])
# Compiling the model with Adamax Optimizer
# model.compile(optimizer='adamax',loss='binary_crossentropy',metrics='accuracy')

# Creating the callback feature to stop the training in-Between, in case of no improvement
# call=callbacks.EarlyStopping(patience=20,min_delta=0.0001,restore_best_weights=True)
# Fitting the model to the training data
# history=model.fit(xtr,ytr,validation_data=(xval,yval),batch_size=28,epochs=150,callbacks=[call])

# Testing on the testing data
# model.evaluate(xte,yte)
# training=pd.DataFrame(history.history)
# training.loc[:,['loss','val_loss']].plot()
# training.loc[:,['accuracy','val_accuracy']].plot()