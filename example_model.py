#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import (
    metrics, 
    preprocessing, 
    linear_model, 
    neural_network, 
    tree, 
    ensemble,
    svm
)
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Lambda
from keras.models import (
    Model,
    Sequential
)
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    ELU,
)

NORMALIZE = not True
INPUT_SHAPE = (50,)

# define base mode
def baseline_model():
    non_linear = 'relu'
    dropout = True
    drop_prob = 0.1
    fc = False

    # create model
    model = Sequential()
    if NORMALIZE:
        model.add(Lambda(lambda x: x/0.5 - 1,
                  input_shape=INPUT_SHAPE))
    model.add(Dense(50, input_dim=50, init='normal'))

    if non_linear: model.add(Activation(non_linear))
    if dropout: model.add(Dropout(drop_prob))
    model.add(Dense(50, init='normal'))
    if non_linear: model.add(Activation(non_linear))
    if dropout: model.add(Dropout(drop_prob))
    model.add(Dense(50, init='normal'))
    if non_linear: model.add(Activation(non_linear))
    if dropout: model.add(Dropout(drop_prob))
    model.add(Dense(50, init='normal'))
    if non_linear: model.add(Activation(non_linear))
    if dropout: model.add(Dropout(drop_prob))
    model.add(Dense(50, init='normal'))
    if non_linear: model.add(Activation(non_linear))
    if dropout: model.add(Dropout(drop_prob))

    if fc:
        model.add(Dense(50, init='normal'))
        model.add(Dense(50, init='normal'))
        model.add(Dense(50, init='normal'))
        model.add(Dense(50, init='normal'))
        model.add(Dense(50, init='normal'))
    
    model.add(Dense(1,  init='normal'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    # autosave best Model and load any previous weights
    model_file = "./model.h5"
    checkpointer = ModelCheckpoint(model_file,
                                   verbose = 1, save_best_only = False)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target']
    X = training_data.drop('target', axis=1)
    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    model = baseline_model()

    print("Training...")
    # Your model is trained on the numerai_training_data
    #model.fit(X, Y)
    X = X.as_matrix(); Y = Y.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(
         X, Y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train, batch_size=100, nb_epoch=25, verbose=1)


    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    #y_prediction = model.predict_proba(X)
    y_prediction = model.predict(X_train)
    print("Log loss on training data is: %s" % (metrics.log_loss(y_train, y_prediction),))
    y_prediction = model.predict(X_test)
    print("Log loss on testing  data is: %s" % (metrics.log_loss(y_test, y_prediction),))  

    y_prediction = model.predict_proba(x_prediction.as_matrix())
    results = y_prediction[:, 0]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
