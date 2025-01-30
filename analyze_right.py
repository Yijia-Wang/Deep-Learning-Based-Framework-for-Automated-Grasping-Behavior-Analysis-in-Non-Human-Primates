import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def analyze_movements(csv_file, smooth=True, window_length=5, polyorder=2):
    df = pd.read_csv(csv_file)
    df = df.fillna(0)

    if 'time' not in df.columns:
        fps = 30 
        df['time'] = df.index / fps

    column_mapping = {
        'x8': None, 'y8': None, 'z8': None,
        'x4': None, 'y4': None, 'z4': None
    }
    for col in df.columns:
        normalized_col = col.replace('_', '').lower()  
        if normalized_col in column_mapping:
            column_mapping[normalized_col] = col

    if any(value is None for value in column_mapping.values()):
        missing_cols = [key for key, value in column_mapping.items() if value is None]
        raise ValueError(f"missing column{', '.join(missing_cols)}，please check your csv")

    x8, y8, z8 = column_mapping['x8'], column_mapping['y8'], column_mapping['z8']
    x4, y4, z4 = column_mapping['x4'], column_mapping['y4'], column_mapping['z4']

    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    
    df['distance'] = np.sqrt((df[x8] - df[x4])**2 +
                             (df[y8] - df[y4])**2 +
                             (df[z8] - df[z4])**2)

    df['velocity'] = np.gradient(df['distance'], df['time'])

    df['acceleration'] = np.gradient(df['velocity'], df['time'])

    if smooth and len(df['distance']) >= window_length:
        df['distance_smooth'] = savgol_filter(
            df['distance'], window_length=window_length, polyorder=polyorder)
        df['velocity_smooth'] = savgol_filter(
            df['velocity'], window_length=window_length, polyorder=polyorder)
        df['acceleration_smooth'] = savgol_filter(
            df['acceleration'], window_length=window_length, polyorder=polyorder)
    else:
        df['distance_smooth'] = df['distance']
        df['velocity_smooth'] = df['velocity']
        df['acceleration_smooth'] = df['acceleration']

    return df


def plot_3d_trajectory(df, output_image):
    start_idx = 0
    end_idx = len(df) - 1
    farthest_idx = df['distance_smooth'].idxmax()

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1, projection='3d')

    ax.plot(df['distance_smooth'], df['velocity_smooth'], df['acceleration_smooth'],
            label='3D Trajectory', color='blue')

    ax.scatter(df.loc[start_idx, 'distance_smooth'], 
               df.loc[start_idx, 'velocity_smooth'], 
               df.loc[start_idx, 'acceleration_smooth'],
               color='green', label='Start Point', s=50)

    ax.scatter(df.loc[end_idx, 'distance_smooth'], 
               df.loc[end_idx, 'velocity_smooth'], 
               df.loc[end_idx, 'acceleration_smooth'],
               color='red', label='End Point', s=50)

    ax.scatter(df.loc[farthest_idx, 'distance_smooth'], 
               df.loc[farthest_idx, 'velocity_smooth'], 
               df.loc[farthest_idx, 'acceleration_smooth'],
               color='orange', label='Farthest Point', s=50)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Acceleration')
    ax.set_title('3D Trajectory of Movement')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()


def batch_process(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for csv_file in os.listdir(input_folder):
        if csv_file.endswith(".csv"):
            file_path = os.path.join(input_folder, csv_file)
            try:
                df = analyze_movements(file_path)
                output_image = os.path.join(output_folder, f"{os.path.splitext(csv_file)[0]}_3d_trajectory.png")
                plot_3d_trajectory(df, output_image)

                print(f"succeed{csv_file}")
            except Exception as e:
                print(f"not succeed {csv_file} with error:{e}")


if __name__ == "__main__":
    input_folder = "right_output"  
    output_folder = "right_output_trajectory" 

    batch_process(input_folder, output_folder)
