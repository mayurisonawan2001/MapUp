import pandas as pd
import numpy as np
from typing import List

# Question 9: Distance Matrix Calculation
def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(MapUp/dataset-2.csv)

    unique_ids = set(df['id_from']).union(set(df['id_to']))
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        id_from = row['id_from']
        id_to = row['id_to']
        distance = row['distance']

        distance_matrix.at[id_from, id_to] = distance
        distance_matrix.at[id_to, id_from] = distance  # Ensure symmetry

    np.fill_diagonal(distance_matrix.values, 0)

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

# Question 10: Unroll Distance Matrix
def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []

    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same ID pairs
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

# Question 11: Finding IDs within Percentage Threshold
def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> List[int]:
    
    reference_distances = df[df['id_start'] == reference_id]['distance']
    average_distance = reference_distances.mean()

    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find all IDs within the range
    ids_within_threshold = df[(df['id_start'] != reference_id) & 
                               (df['distance'] >= lower_bound) & 
                               (df['distance'] <= upper_bound)]

    return sorted(ids_within_threshold['id_start'].unique().tolist())

# Question 12: Calculate Toll Rate
def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
   
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df

# Question 13: Calculate Time-Based Toll Rates
def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
   
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Expand the DataFrame with time and day information
    expanded_data = []
    for _, row in df.iterrows():
        for day in days_of_week:
            # Simulate all time slots
            time_slots = [
                ('00:00:00', '10:00:00', 0.8),
                ('10:00:00', '18:00:00', 1.2),
                ('18:00:00', '23:59:59', 0.8)
            ]
            if day in ['Saturday', 'Sunday']:
                time_slots = [(slot[0], slot[1], 0.7) for slot in time_slots]  # Apply discount for weekends
            
            for start_time, end_time, discount in time_slots:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = pd.to_datetime(start_time).time()
                new_row['end_time'] = pd.to_datetime(end_time).time()

                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    new_row[vehicle] *= discount
                
                expanded_data.append(new_row)

    time_based_df = pd.DataFrame(expanded_data)
    
    return time_based_df


file_path_2 = 'MapUp/dataset-2.csv'
distance_matrix = calculate_distance_matrix(file_path_2)

# Unroll the distance matrix for Question 10
unrolled_df = unroll_distance_matrix(distance_matrix)

# Finding IDs within percentage threshold for Question 11
reference_id = 1  # Replace with the desired reference ID
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Calculate toll rates for Question 12
toll_rate_df = calculate_toll_rate(unrolled_df)

# Calculate time-based toll rates for Question 13
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)

print("Unrolled DataFrame:\n", unrolled_df.head())
print("IDs within 10% threshold of reference ID:", ids_within_threshold)
print("Toll Rates DataFrame:\n", toll_rate_df.head())
print("Time-based Toll Rates DataFrame:\n", time_based_toll_df.head())
