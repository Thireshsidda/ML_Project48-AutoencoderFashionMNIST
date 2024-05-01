# ML_Project48-AutoencoderFashionMNIST

### Convolutional Autoencoder for Fashion MNIST Classification
This repository implements a convolutional autoencoder for image classification on the Fashion MNIST dataset. The code leverages the encoder's ability to learn useful features from the data for effective classification.

### Project Overview

Data: Fashion MNIST dataset containing 70,000 grayscale images of various clothing items.

Model: Convolutional autoencoder with batch normalization and ReLU activation.

Objective: Train the autoencoder to reconstruct the input images and extract features for classification.

Classification: Utilize the encoder part of the trained autoencoder as a feature extractor for a new model with fully-connected layers for classifying fashion items.


### Key Points

#### Data Preprocessing:
Efficient data loading and reshaping for compatibility with convolutional layers.

Pixel value normalization between 0 and 1.

Splitting data into training and validation sets.

#### Autoencoder Architecture:

Encoder: Convolutional layers with batch normalization and ReLU activation.

Decoder: UpSampling2D layers for image reconstruction, convolutional layers with ReLU and sigmoid activation for final output.

Compiled with RMSprop optimizer and mean squared error loss.

#### Training and Evaluation:

Training for 200 epochs using the fit function.

Visualizing training and validation loss curves to monitor performance and avoid overfitting.

#### Classification with Encoder:
Extracting encoder weights from the trained autoencoder.

One-hot encoding of categorical labels.

Splitting data again for classification.

Building a new model with encoder as input, followed by fully-connected layers with ReLU and Softmax activation.

Setting encoder weights to the trained autoencoder weights for efficient feature extraction.

### Benefits

Leverages autoencoder's feature learning capability for improved classification.

Batch normalization reduces overfitting during training.

Visualization of loss curves provides valuable insights into training progress.

One-hot encoding ensures proper handling of categorical labels.

### Running the Project

Install required libraries (e.g., TensorFlow, Keras).

Run the script to train the autoencoder and perform classification.

Observe the loss curves and classification accuracy.

### Further Exploration

Experiment with different hyperparameters (e.g., number of layers, units per layer) to optimize performance.

Implement different autoencoder architectures (e.g., variational autoencoders).

Explore alternative classification models using the learned feature
