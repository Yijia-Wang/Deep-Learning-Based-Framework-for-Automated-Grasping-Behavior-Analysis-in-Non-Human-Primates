import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def analyze_hand_movement(csv_file, output_image, fps=30, smooth=True, window_length=5, polyorder=2):
    df = pd.read_csv(csv_file)

    column_mapping = {
        'x0': None, 'y0': None, 'z0': None 
    }
    for col in df.columns:
        normalized_col = col.replace('_', '').lower()
        if normalized_col in column_mapping:
            column_mapping[normalized_col] = col

    if any(value is None for value in column_mapping.values()):
        missing_cols = [key for key, value in column_mapping.items() if value is None]
        raise ValueError(f"missing colmn：{', '.join(missing_cols)}，please check your csv file！")

    x, y, z = column_mapping['x0'], column_mapping['y0'], column_mapping['z0']

    df = df.interpolate(method='linear', limit_direction='forward', axis=0)

    if 'time' not in df.columns:
        df['time'] = df.index / fps 

    df['distance'] = 100 * np.sqrt((df[x].diff())**2 + (df[y].diff())**2 + (df[z].diff())**2)
    df['distance'].fillna(0, inplace=True)

    df['velocity'] = df['distance'].diff() / df['time'].diff()
    df['velocity'].fillna(0, inplace=True)

    df['acceleration'] = df['velocity'].diff() / df['time'].diff()
    df['acceleration'].fillna(0, inplace=True)

    if smooth and len(df) >= window_length:
        df['distance_smooth'] = savgol_filter(df['distance'], window_length=window_length, polyorder=polyorder)
        df['velocity_smooth'] = savgol_filter(df['velocity'], window_length=window_length, polyorder=polyorder)
        df['acceleration_smooth'] = savgol_filter(df['acceleration'], window_length=window_length, polyorder=polyorder)
    else:
        df['distance_smooth'] = df['distance']
        df['velocity_smooth'] = df['velocity']
        df['acceleration_smooth'] = df['acceleration']

    peak_velocity = df['velocity_smooth'].max() 
    avg_velocity = df['velocity_smooth'].mean()  
    total_distance = df['distance_smooth'].sum() 
    peak_velocity_time = df.loc[df['velocity_smooth'].idxmax(), 'time'] 

    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['distance_smooth'], label="Distance (Smoothed)", color='blue')
    plt.title("Overall Hand Movement Distance Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance (cm)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['velocity_smooth'], label="Velocity (Smoothed)", color='orange')
    plt.axvline(peak_velocity_time, color='green', linestyle="--", label="Peak Velocity Time")
    plt.title("Overall Hand Movement Velocity Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (cm/s)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['acceleration_smooth'], label="Acceleration (Smoothed)", color='red')
    plt.title("Overall Hand Movement Acceleration Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (cm/s²)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

    return peak_velocity, avg_velocity, total_distance, peak_velocity_time


def batch_process(input_folder, output_folder, fps=30, smooth=True, window_length=5, polyorder=2):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_image = os.path.join(output_folder, f"{file_name}_overall_movement.png")

        try:
            print(f"Processing: {csv_file}")
            peak_velocity, avg_velocity, total_distance, peak_velocity_time = analyze_hand_movement(
                csv_file, output_image, fps, smooth, window_length, polyorder
            )
            results.append([file_name, peak_velocity, avg_velocity, total_distance, peak_velocity_time])
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    result_df = pd.DataFrame(results, columns=['File', 'Peak Velocity (cm/s)', 'Avg Velocity (cm/s)',
                                               'Total Distance (cm)', 'Peak Velocity Time (s)'])
    result_df.to_csv(os.path.join(output_folder, "summary_overhead_movement.csv"), index=False)

    print("Processing completed. Results saved.")


if __name__ == "__main__":
    input_folder = "overhead_output"  
    output_folder = "overhead_output_p" 

    batch_process(input_folder, output_folder, fps=30, smooth=True, window_length=11, polyorder=3)
