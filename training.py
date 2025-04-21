"""
Author: Giovanni Lucente

This script processes the NGSIM dataset, prepares training and testing data for a neural network, 
trains an LSTM-based model, and evaluates its performance in predicting vehicle longitudinal motion.
"""

# Import required libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Load dataset
file_path = "NGSIM_data.pkl"
with open(file_path, "rb") as file:
    X = pickle.load(file)  # Dictionary format: X[vehicle_id][frame] = (x, y, speed, acceleration)

# Define parameters
num_timesteps = 101   # Total number of timesteps per sample
num_timesteps_input = 30  # Number of timesteps used as input
num_timesteps_output = num_timesteps  # Number of timesteps to predict

# Calculate total number of training samples
tot_samples = sum(
    int((len(frames) - num_timesteps) / (0.1 * num_timesteps))
    for frames in (list(X[vid].keys()) for vid in X)
)

# Initialize position array
position = np.zeros((tot_samples, num_timesteps, 3))  # Stores distance, speed, acceleration
index = 0

# Process data for each vehicle
for vehicle_id, frames in X.items():
    frame_list = list(frames.keys())

    for i in range(int((len(frame_list) - num_timesteps) / (0.1 * num_timesteps))):
        start_frame = i * int(0.1 * num_timesteps)

        x_ego, y_ego = [], []
        for t in range(num_timesteps):
            x_ego.append(X[vehicle_id][start_frame + t][0])
            y_ego.append(X[vehicle_id][start_frame + t][1])

            position[index, t, 1] = X[vehicle_id][start_frame + t][2]  # Speed
            position[index, t, 2] = X[vehicle_id][start_frame + t][3]  # Acceleration

        # Compute cumulative distance traveled
        dx = np.append([0], np.diff(x_ego))
        dy = np.append([0], np.diff(y_ego))
        position[index, :, 0] = np.cumsum(np.sqrt(dx**2 + dy**2))

        index += 1  # Move to next sample

# Prepare input (X) and output (y) arrays
final_position = np.zeros((index, num_timesteps_output, 2))
final_position[:, :, 0] = position[:, :num_timesteps_output, 0]  # Distance
final_position[:, :, 1] = position[:, :num_timesteps_output, 1]  # Speed
input_position = position[:, :num_timesteps_input, :]

# Split data into training, validation, and test sets
train_ratio, test_ratio = 0.7, 0.1
num_samples = input_position.shape[0]

train_size = int(num_samples * train_ratio)
test_size = int(num_samples * test_ratio)

train_indices = np.random.choice(num_samples, train_size, replace=False)
X_train, y_train = input_position[train_indices], final_position[train_indices]

remaining_indices = np.setdiff1d(np.arange(num_samples), train_indices)
X_remaining, y_remaining = input_position[remaining_indices], final_position[remaining_indices]

test_indices = np.random.choice(X_remaining.shape[0], test_size, replace=False)
X_test, y_test = X_remaining[test_indices], y_remaining[test_indices]

X_validation = np.delete(X_remaining, test_indices, axis=0)
y_validation = np.delete(y_remaining, test_indices, axis=0)

# Define LSTM-based Neural Network
model = Sequential([
    LSTM(32, activation='tanh', return_sequences=True),
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(num_timesteps_output * 2, activation='relu'),
    Reshape((num_timesteps_output, 2))
])

# Define model checkpoint callback
model_checkpoint_path = 'C:\\Users\\luce_gi\\Desktop\\ML\\model\\'
cp = ModelCheckpoint(model_checkpoint_path, save_best_only=True)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=1.0),
              loss='mean_squared_error', metrics=['mse'])

# Train model
hist = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                 epochs=150, batch_size=256, verbose=1, callbacks=[cp])

# Model summary
model.summary()

# Load the best model
best_model = load_model(model_checkpoint_path)
y_pred = best_model.predict(X_test)

# Evaluate model performance at different horizons
for seconds in range(1, 6):
    eval_horizon = num_timesteps_input + (seconds * 10)
    mse = mean_squared_error(y_test[:, eval_horizon, 0], y_pred[:, eval_horizon, 0])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test[:, eval_horizon, 0], y_pred[:, eval_horizon, 0])
    
    print(f"\nRoot Mean Squared Error ({seconds}s): {rmse:.4f}")
    print(f"Mean Absolute Error ({seconds}s): {mae:.4f}")

# Plot predicted vs actual results for the last timestep
plt.plot(range(y_pred.shape[0]), y_pred[:, -1, 0], label="Predicted")
plt.plot(range(y_test.shape[0]), y_test[:, -1, 0], label="Actual")
plt.legend()
plt.title("Predicted vs Actual Progression")
plt.show()

# Randomly visualize 10 test cases where speed > 2.0
np.random.seed(np.random.randint(0, 73))
count = 0

for _ in range(100):
    sample_index = np.random.randint(0, X_test.shape[0])
    x_axis = np.arange(y_test.shape[1])

    if y_test[sample_index, -1, 1] > 2.0:
        plt.plot(x_axis, y_test[sample_index, :, 0], marker=".", label="True Progress")
        plt.plot(x_axis, y_pred[sample_index, :, 0], marker=".", label="Predicted Progress")
        plt.legend()
        plt.axis('equal')
        plt.show()
        
        count += 1
        if count == 10:
            break

# Plot training history
plt.plot(hist.history['mse'], label="Train MSE")
plt.plot(hist.history['val_mse'], label="Validation MSE")
plt.title("Model MSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
