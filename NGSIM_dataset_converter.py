import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os

# The objective of this file is to convert the raw NGSIM dataset into a format that can be used for training.
# The format consists of a list X in which:
# X[ID][frame][0] = x coordinate of the vehicle with ID at a specific frame
# X[ID][frame][1] = y coordinate of the vehicle with ID at a specific frame
# X[ID][frame][2] = speed of the vehicle with ID at a specific frame
# X[ID][frame][3] = acceleration of the vehicle with ID at a specific frame
# X[ID][frame][4] = space headway (distance to front vehicle) of the vehicle with ID at a specific frame


def load_data(file_paths, conversion_coefficient=0.3048, start_id=0):
    X = {}
    first_frame_per_vehicle = {}
    max_id = start_id
    # Columns of the dataset:
    # ID | frame | tot_frames | global_time | loc_x | loc_y | glob_x | glob_y | length | width | class | speed | acceleration | ... | space headway (23^th)
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for line in file:
                columns = line.split()
                id = int(columns[0]) + max_id
                frame = int(columns[1])
                x = conversion_coefficient * float(columns[4])
                y = conversion_coefficient * float(columns[5])
                speed = conversion_coefficient * float(columns[11])
                acc = conversion_coefficient * float(columns[12])
                space_headway = conversion_coefficient * float(columns[17])
                
                if id not in first_frame_per_vehicle:
                    first_frame_per_vehicle[id] = frame
                
                relative_frame = frame - first_frame_per_vehicle[id]
                
                if id not in X:
                    X[id] = {}
                X[id][relative_frame] = (x, y, speed, acc, space_headway)
        
        max_id = max(X.keys(), default=start_id)  # Update max_id for next dataset
    
    return X

def main():
    data_dir = "C:\\Users\\luce_gi\\Desktop\\ML\\NGSIM\\"
    file_names = [
        "i80_trajectories-0400-0415.txt",
        "i80_trajectories-0500-0515.txt",
        "i80_trajectories-0515-0530.txt",
        "i101_trajectories-0750am-0805am.txt",
        "i101_trajectories-0805am-0820am.txt",
        "i101_trajectories-0820am-0835am.txt"
    ]
    
    file_paths = [os.path.join(data_dir, name) for name in file_names]
    X = load_data(file_paths)
    
    print("Number of IDs contained in X:", len(X))
    
    random_vehicle_id = random.choice(list(X.keys()))
    frames = list(X[random_vehicle_id].keys())
    speeds = [X[random_vehicle_id][frame][2] for frame in frames]
    accelerations = [X[random_vehicle_id][frame][3] for frame in frames]
    x_positions = [X[random_vehicle_id][frame][0] for frame in frames]
    y_positions = [X[random_vehicle_id][frame][1] for frame in frames]
    
    frame_lengths = np.array([len(vehicle) for vehicle in X.values()])
    
    print(f"Mean number of frames per vehicle: {np.mean(frame_lengths)}")
    print(f"Standard deviation of frames per vehicle: {np.std(frame_lengths)}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(frames, speeds, label='Speed')
    plt.plot(frames, accelerations, label='Acceleration')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title(f'Speed and Acceleration Profile for Vehicle ID {random_vehicle_id}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.scatter(x_positions, y_positions, label='trajectory', color='red', marker='o')
    plt.xlabel('x_positions')
    plt.ylabel('y_positions')
    plt.title(f'X,Y Position Profile for Vehicle ID {random_vehicle_id}')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save the dictionary to a file using pickle
    with open("NGSIM_data.pkl", "wb") as file:
        pickle.dump(X, file)
    
if __name__ == "__main__":
    main()
