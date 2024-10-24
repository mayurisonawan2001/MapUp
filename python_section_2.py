import pandas as pd
import numpy as np
import datetime

def calculate_distance_matrix(df)->pd.DataFrame():
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


def unroll_distance_matrix(df)->pd.DataFrame():
    unrolled_data = []

    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same ID pairs
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    reference_distances = df[df['id_start'] == reference_id]['distance']

    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])

    average_distance = reference_distances.mean()

    lower_bound = average_distance * 0.90
    upper_bound = average_distance * 1.10

    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    average_distances.columns = ['id_start', 'average_distance']

    ids_within_threshold = average_distances[(average_distances['average_distance'] >= lower_bound) & 
                                             (average_distances['average_distance'] <= upper_bound)]

    ids_within_threshold = ids_within_threshold.sort_values(by='id_start')

    return ids_within_threshold



def calculate_toll_rate(df)->pd.DataFrame():
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
   weekday_discount = {
        "morning": 0.8,  
        "day": 1.2,     
        "evening": 0.8   
    }
    weekend_discount = 0.7 

    time_intervals = {
        "morning": (datetime.time(0, 0), datetime.time(10, 0)),
        "day": (datetime.time(10, 0), datetime.time(18, 0)),
        "evening": (datetime.time(18, 0), datetime.time(23, 59, 59))
    }
    
    df['start_day'] = ''
    df['end_day'] = ''
    df['start_time'] = datetime.time(0, 0)  # Default start time
    df['end_time'] = datetime.time(23, 59, 59)  # Default end time

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for index, row in df.iterrows():
        start_day_index = row['id_start'] % 7
        end_day_index = row['id_end'] % 7
        
        row['start_day'] = days_of_week[start_day_index]
        row['end_day'] = days_of_week[end_day_index]

        if row['start_day'] in days_of_week[:5]:  # Monday to Friday
            for period, (start_time, end_time) in time_intervals.items():
                if start_time <= row['start_time'] < end_time:
                    discount_factor = weekday_discount[period]
                    break
        else:  
            discount_factor = weekend_discount

        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] *= discount_factor
        df.loc[index] = row

    return df
