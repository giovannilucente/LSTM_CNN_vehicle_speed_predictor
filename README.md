# Vehicle speed predictor based on LSTM-CNN hybrid neural network architecture

This repository contains code for training a model that predicts vehicle speed using a LSTM-CNN hybrid network architecture. Follow the steps below to set up and run the training process.
For more information visit the [project page](https://giovannilucente.github.io/LSTM_CNN_vehicle_speed_predictor/index.html). For further information please refer to the related paper:

Giovanni Lucente, Mikkel Skov Maarssoe, Iris Kahl, Julian Schindler, **Deep Learning Algorithms for Longitudinal Driving Behavior Prediction: A Comparative Analysis of Convolutional Neural Network and Long–Short-Term Memory Models**, *SAE International Journal of Connected and Automated Vehicles*, 2024. DOI: [10.4271/12-07-04-0025](https://doi.org/10.4271/12-07-04-0025).

## Installation and Setup

To install and run this project locally, follow these steps:

### 1. Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/giovannilucente/LSTM_CNN_vehicle_speed_predictor.git
cd LSTM_CNN_vehicle_speed_predictor
```
### 2. Install the requirements
Install all the required dependencies listed in the requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Download the NGSIM dataset
Navigate to the NGSIM folder and download the NGSIM trajectory dataset (in .txt format) from the official site:
```bash
cd NGSIM
# Download here the trajectory dataset in .txt format
cd ..
```
Note: Please ensure that you download the dataset into the NGSIM folder.

### 4. Convert the dataset
Once the dataset is downloaded, use the provided Python script to convert the raw NGSIM dataset into a format that can be used for training. Run the following command:
```bash
python3 NGSIM_dataset_converter.py
```
A file called NGSIM_data.pkl will be created, containing the converted dataset in a Python pickle format.

### 5. Launch the training
Now you can train the model using the following command:
```bash
python3 training.py
```
The model will train, and the best-performing model will be saved in a folder called model.
Some representative results are the following:
<p align="center">
  <img src="media/ex3.png" alt="ex3" width="45%"/>
  <img src="media/ex7.png" alt="ex7" width="45%"/>
</p>
