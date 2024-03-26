# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 18:55:44 2023

@author: Sheilla

Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""

# Import required libraries
import os
import numpy as np 
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Testing a Convolutional Neural Network (CNN) model for a keypoint detection task.')

# Add the required arguments
parser.add_argument('dataset', type=str, help='Path to the dataset directory')
parser.add_argument('model_path', type=str, help='Path to the pretrained model .h5 file')
parser.add_argument('save_path', type=str, help='Path to save the output files')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
dataset_dir = args.dataset
model_dir = args.model_path
save_path = args.save_path

# Update the code to use the values of the arguments
# Load the feature maps and labels for testing
featuremap_test = np.load(os.path.join(dataset_dir, 'unsorted_only-xyz_test.npy'))
labels_test = np.load(os.path.join(dataset_dir, 'labels_test.npy'))

# Print the shape of the feature maps and labels
print('Feature maps for testing:', featuremap_test.shape)
print('Labels for testing:', labels_test.shape)

# Load the model
model = tf.keras.models.load_model(model_dir)

# Evaluate the model on the test set
score_test = model.evaluate(featuremap_test, labels_test, verbose=1)

# Print the evaluation results
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])

# Predict the keypoints on the test set
result_test = model.predict(featuremap_test)

# Calculate the localization error
error = np.sqrt(np.sum((labels_test - result_test)**2, axis=1))

# Save the localization error to a file
np.save(os.path.join(save_path, 'localization_error.npy'), error)

# Plot the histogram of the localization error
plt.hist(error, bins=50, color='blue', edgecolor='black')
plt.xlabel('Localization Error')
plt.ylabel('Frequency')
plt.title('Histogram of Localization Error')
plt.savefig(os.path.join(save_path, 'localization_error.png'))
plt.show()

# Calculate the mean localization error
mean_error = np.mean(error)

# Print the mean localization error
print('Mean Localization Error:', mean_error)

# Save the mean localization error to a file
with open(os.path.join(save_path, 'mean_localization_error.txt'), 'w') as f:
    f.write('Mean Localization Error: ' + str(mean_error))
    
# Save the predicted keypoints to a file
np.save(os.path.join(save_path, 'predicted_keypoints.npy'), result_test)

# plot the 3d points
fig = plt.figure(figsize=plt.figaspect(0.5))

# load points for plotting 3D figure
# for i in range(len(labels_test)):
for i in range(0,100):
    data_labels = labels_test[i]
    data_results = result_test[i]

    body_data_x_labels = np.array(data_labels[0 : 19], dtype=float)
    body_data_y_labels = np.array(data_labels[19 : 38], dtype=float)
    body_data_z_labels = np.array(data_labels[38 : ], dtype=float)

    body_data_x_results = np.array(data_results[0 : 19], dtype=float)
    body_data_y_results = np.array(data_results[19 : 38], dtype=float)
    body_data_z_results = np.array(data_results[38 : ], dtype=float)

    # plot the 3d points
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.clear()

    ax.scatter(body_data_x_labels, body_data_y_labels, body_data_z_labels, color='blue', marker='o', linewidths=2)
    ax.scatter(body_data_x_results, body_data_y_results, body_data_z_results, color='red', marker='o', linewidths=2)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 3)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X (Azimuth)')
    ax.set_ylabel('Z (Range)')
    ax.set_zlabel('Y (Elevation)')

    ax.set_title('3D Scatter Plot of Predicted Keypoints')

    plt.savefig(os.path.join(save_path, f'3d_scatter_plot_{i}.png'))

# Close the plot at the end
plt.close(fig)