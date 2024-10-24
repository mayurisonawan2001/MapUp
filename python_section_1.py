import re
from typing import Dict, List
import polyline
import numpy as np
import pandas as pd
from typing import List
from typing import Any, Dict


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []  
    length = len(lst)
    
    for i in range(0, length, n):
        group = []  # To store the current group
        
        for j in range(min(n, length - i)):  
            group.append(lst[i + (n - 1 - j)])  

        result.extend(group)  
    lst = result
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string) 
        
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(d: Dict[Any, Any], parent_key: str = '') -> Dict:
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(_flatten(value, new_key).items())
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    list_key = f"{new_key}[{index}]"
                    items.extend(_flatten({list_key: item}, parent_key).items())
            else:
                items.append((new_key, value))
        return dict(items)
     return _flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])  
            return
        
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            used[i] = True
            path.append(nums[i])
            
            backtrack(path, used)
            
            path.pop()
            used[i] = False

    nums.sort()  
    result = []
    used = [False] * len(nums) 
    backtrack([], used) 
    return result


def find_all_dates(text: str) -> List[str]:
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b', 
        r'\b\d{4}\.\d{2}\.\d{2}\b'
    ]
    combined_pattern = '|'.join(patterns)
    dates = re.findall(combined_pattern, text)
    return dates
    pass

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371000  
        return c * r

    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'], df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)
    
    df['distance'] = distances
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)

    rotated_matrix = [[0] * n for _ in range(n)] 
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    transformed_matrix = [[0] * n for _ in range(n)]  
    for i in range(n):
        for j in range(n):
            original_row = n - 1 - j  
            original_col = i          
            index_sum = original_row + original_col  
            transformed_matrix[i][j] = rotated_matrix[i][j] * index_sum  

    return transformed_matrix


df = pd.read_csv('MapUp/dataset-1.csv')
results = time_check(df)
def time_check(df) -> pd.Series:
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    grouped = df.groupby(['id', 'id_2'])
   
    def check_group(group):
        full_days = pd.date_range(start='2022-01-01', periods=7, freq='D')
        
        present_days = group['start_datetime'].dt.date.unique()
        
        for day in full_days:
            if day.date() in present_days:
                day_start = pd.Timestamp(day.date())  # 12:00 AM
                day_end = pd.Timestamp(day.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)  # 11:59:59 PM
                
                daily_group = group[(group['start_datetime'].dt.date == day.date()) |
                                    (group['end_datetime'].dt.date == day.date())]
                
                if not (daily_group['start_datetime'].min() <= day_start and daily_group['end_datetime'].max() >= day_end):
                    return True  # Incomplete for this day

        return False  # Complete for this (id, id_2)

    completeness_results = grouped.apply(check_group)
    
    return pd.Series(completeness_results, index=completeness_results.index)
