# PlantGuard

## Plant Disease Detection using Convolutional Neural Networks (CNNs)

This project aims to develop a deep learning model for detecting plant diseases based on leaf images. The model is built using TensorFlow and Keras, and it utilizes a convolutional neural network (CNN) architecture to classify leaf images into different disease categories.

## Dataset

The dataset used in this project consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is divided into an 80/20 ratio for training and validation, and a separate directory containing 33 test images is used for prediction purposes.

## Model Architecture

The model architecture consists of several convolutional and pooling layers, followed by a flattening layer and a dense layer with dropout regularization. The final layer is a dense layer with softmax activation for multi-class classification.

## Model Training

The model is trained using the Adam optimizer with a categorical cross-entropy loss function and accuracy as the evaluation metric. The training process is monitored using TensorBoard.

## Model Evaluation

The model is evaluated using classification metrics such as precision, recall, and F1-score, as well as a confusion matrix.

## Model Deployment

The trained model is deployed using Streamlit, a popular open-source app framework for machine learning and data science applications. The app allows users to upload leaf images and receive predictions for the corresponding plant diseases.

## Requirements

The following packages are required to run this project:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorBoard
- Streamlit

## Usage

To run the project, simply clone the repository and install the required packages using pip. Then, run the Streamlit app using the command:


streamlit run app.py

## Contributing

Contributions to this project are welcome! If you have any suggestions or improvements, please submit a pull request.


## Acknowledgments

The original dataset used in this project is from the Kaggle competition "Plant Disease Detection".
