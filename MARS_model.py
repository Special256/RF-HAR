# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 18:55:44 2023

@author: Sheilla
"""

"""
This script trains a Convolutional Neural Network (CNN) model for a keypoint detection task.
It uses point cloud feature maps and labels as input, and evaluates the model's performance on a test set.

Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""
# Import required libraries

import os
import argparse
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Train a Convolutional Neural Network (CNN) model for a keypoint detection task.')

# Add the required arguments
parser.add_argument('dataset', type=str, help='Path to the dataset directory')
parser.add_argument('model_dir', type=str, help='Path to the model directory')
parser.add_argument('save_path', type=str, help='Path to save the output files')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
dataset_dir = args.dataset
model_dir = args.model_dir
save_path = args.save_path

# Update the code to use the values of the arguments
# Load the feature maps and labels for training, validation, and testing
featuremap_train = np.load(os.path.join(dataset_dir, 'unsorted_only-xyz_train.npy'))
featuremap_validate = np.load(os.path.join(dataset_dir, 'unsorted_only-xyz_validate.npy'))
featuremap_test = np.load(os.path.join(dataset_dir, 'unsorted_only-xyz_test.npy'))

labels_train = np.load(os.path.join(dataset_dir, 'labels_train.npy'))
labels_validate = np.load(os.path.join(dataset_dir, 'labels_validate.npy'))
labels_test = np.load(os.path.join(dataset_dir, 'labels_test.npy'))

# Print the shape of the feature maps and labels
print('Feature maps for training:', featuremap_train.shape)
print('Feature maps for validation:', featuremap_validate.shape)
print('Feature maps for testing:', featuremap_test.shape)

# Initialize the result array
paper_result_list = []

# Define batch size and epochs
batch_size = 128
epochs = 200

# Define the CNN model architecture
def define_CNN(in_shape, n_keypoints):
    in_one = Input(shape=in_shape)
    conv_one_1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(in_one)
    conv_one_1 = Dropout(0.3)(conv_one_1)
    conv_one_1 = BatchNormalization(momentum=0.95)(conv_one_1)
    
    conv_one_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(conv_one_1)
    conv_one_2 = Dropout(0.3)(conv_one_2)
    conv_one_2 = BatchNormalization(momentum=0.95)(conv_one_2)

    fe = Flatten()(conv_one_2)
    dense_layer1 = Dense(512, activation='relu')(fe)
    dense_layer1 = BatchNormalization(momentum=0.95)(dense_layer1)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    
    out_layer = Dense(n_keypoints, activation='linear')(dense_layer1)

    model = Model(in_one, out_layer)
    opt = Adam(lr=0.001, beta_1=0.5)

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    return model

history = []

# Repeat the training process for multiple iterations
for i in range(10):
    # Instantiate the model
    keypoint_model = define_CNN(featuremap_train[0].shape, 57) # 57 = 19 body joints with 3 axis each(x,y,z)
    score_min = 10  # Initial maximum error

    # Train the model
    history = keypoint_model.fit(featuremap_train, labels_train,
                                 batch_size=batch_size, epochs=epochs, verbose=1, 
                                 validation_data=(featuremap_validate, labels_validate))

    # Evaluate the model on training and validation sets
    score_train = keypoint_model.evaluate(featuremap_train, labels_train, verbose=1)
    score_validate = keypoint_model.evaluate(featuremap_validate, labels_validate, verbose=1)
    print('train metrics = ', score_train)
    print('validate metrics = ', score_validate)

    # Predict keypoints on the test set
    result_test = keypoint_model.predict(featuremap_test)

    # Calculate mean absolute error (MAE) for each axis
    print("mae for x is", metrics.mean_absolute_error(labels_test[:, 0:19], result_test[:, 0:19]))
    print("mae for y is", metrics.mean_absolute_error(labels_test[:, 19:38], result_test[:, 19:38]))
    print("mae for z is", metrics.mean_absolute_error(labels_test[:, 38:57], result_test[:, 38:57]))

    # Calculate mean absolute error (MAE) and root mean squared error (RMSE) for each axis
    x_mae = metrics.mean_absolute_error(labels_test[:, 0:19], result_test[:, 0:19], multioutput='raw_values')
    y_mae = metrics.mean_absolute_error(labels_test[:, 19:38], result_test[:, 19:38], multioutput='raw_values')
    z_mae = metrics.mean_absolute_error(labels_test[:, 38:57], result_test[:, 38:57], multioutput='raw_values')
    
    all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 19)
    avg_19_points_mae = np.mean(all_19_points_mae, axis=0)
    avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis=1).reshape(1, 3)

    # Matrix transformation for the final all 19 points MAE
    all_19_points_mae_Transpose = all_19_points_mae.T
    
    x_rmse = metrics.mean_squared_error(labels_test[:, 0:19], result_test[:, 0:19], multioutput='raw_values', squared=False)
    y_rmse = metrics.mean_squared_error(labels_test[:, 19:38], result_test[:, 19:38], multioutput='raw_values', squared=False)
    z_rmse = metrics.mean_squared_error(labels_test[:, 38:57], result_test[:, 38:57], multioutput='raw_values', squared=False)
    
    all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3, 19)
    avg_19_points_rmse = np.mean(all_19_points_rmse, axis=0)
    avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis=1).reshape(1, 3)

    # Matrix transformation for the final all 19 points RMSE
    all_19_points_rmse_Transpose = all_19_points_rmse.T
    
    all_19_points_maermse_Transpose = np.concatenate((all_19_points_mae_Transpose, all_19_points_rmse_Transpose), axis=1) * 100
    avg_19_points_maermse_Transpose = np.concatenate((avg_19_points_mae_xyz, avg_19_points_rmse_xyz), axis=1) * 100
    
    paper_result_maermse = np.concatenate((all_19_points_maermse_Transpose, avg_19_points_maermse_Transpose), axis=0)
    paper_result_maermse = np.around(paper_result_maermse, 2)
    # reorder the columns to make it xmae, xrmse, ymae, yrmse, zmae, zrmse, avgmae, avgrmse
    paper_result_maermse = paper_result_maermse[:, [0, 3, 1, 4, 2, 5]]

    # append each iterations result
    paper_result_list.append(paper_result_maermse)
    
    # Check if the output directory exists
    output_direct = model_dir
    
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)

    if(score_validate[1] < score_min):
        keypoint_model.save(output_direct + 'model.h5')
        score_min = score_validate[1]

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_path,'accuracy.png'))
plt.clf()

# Plot MAE
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Xval'], loc='upper left')
plt.savefig(os.path.join(save_path,'MAE.png'))
plt.clf()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Xval'], loc='upper left')
plt.xlim([0, 100])
plt.ylim([0, 0.1])
plt.savefig(os.path.join(save_path,'loss.png'))

# Average the result for all iterations
mean_paper_result_list = np.mean(paper_result_list, axis=0)
mean_mae = np.mean(np.dstack((mean_paper_result_list[:, 0], mean_paper_result_list[:, 2], mean_paper_result_list[:, 4])).reshape(20, 3), axis=1)
mean_rmse = np.mean(np.dstack((mean_paper_result_list[:, 1], mean_paper_result_list[:, 3], mean_paper_result_list[:, 5])).reshape(20, 3), axis=1)
mean_paper_result_list = np.concatenate((np.mean(paper_result_list, axis=0), mean_mae.reshape(20, 1), mean_rmse.reshape(20, 1)), axis=1)

# Export the Accuracy
np.save(os.path.join(save_path, 'results.npy'), mean_paper_result_list)
np.savetxt(os.path.join(save_path, 'results.txt'), mean_paper_result_list, fmt='%.2f')
print("The result is saved in the directory: ", save_path)
print("The model is saved in the directory: ", model_dir)