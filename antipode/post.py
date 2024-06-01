import pandas as pd
import scipy
import numpy as np
import tqdm
from . import plotting

def get_quantile_markers(df,q=0.95):
    """
    Get the markers for the rows of a dataframe.

    :param df: GEX means where rows are categories and columns are named features.
    :param q: Quantile of mean to subtract.
    :return: A matrix which has the difference of each gene in each cluster vs the quantile value in all other clusters.
    """
    df_array=df.to_numpy()
    coefs=[]
    for i in tqdm.tqdm(range(df.shape[0])):
        others=list(set(list(range(df.shape[0])))-set([i]))
        coefs.append((df_array[i:(i+1),:]-np.quantile(df_array[others,:],q,axis=0)))#/(cluster_params.std(0)+cluster_params.std(0).mean()))
    coefs=np.concatenate(coefs,axis=0)
    marker_df=pd.DataFrame(coefs,index=df.index,columns=df.columns)
    return(marker_df)

def get_n_largest(n):
    def get_top_n(x):
        return x.nlargest(n).index.tolist()
    return(get_top_n)

def resampling_p_value(data, group_labels,fun, num_iterations=1000):
    """
    Calculate the resampling p-value for the magnitude of input values, partitioned by two groups.
    
    Arguments:
    data -- A list or NumPy array of input values.
    group_labels -- A list or NumPy array of group labels corresponding to each value in the data.
    num_iterations -- The number of iterations to perform for the resampling (default: 1000).
    
    Returns:
    p_value -- The resampling p-value.
    """
    group_labels = np.array(group_labels)
    data = np.array(data)
    group1_data = data[group_labels]
    group2_data = data[~group_labels]
    observed_difference = np.abs(np.mean(fun(group1_data)) - np.mean(fun(group2_data)))

    combined_data = np.concatenate((group1_data, group2_data))
    num_group1 = len(group1_data)
    num_group2 = len(group2_data)
    num_total = num_group1 + num_group2
    larger_difference_count = 0

    for _ in tqdm.tqdm(range(num_iterations)):
        np.random.shuffle(combined_data)
        perm_group1 = combined_data[:num_group1]
        perm_group2 = combined_data[num_group1:]
        perm_difference = np.abs(np.mean(fun(perm_group1)) - np.mean(fun(perm_group2)))
        if perm_difference >= observed_difference:
            larger_difference_count += 1

    p_value = (larger_difference_count + 1) / (num_iterations + 1)
    return p_value

def resampling_slope_p_value(x, y, num_iterations=1000):
    """
    Calculate the resampling p-value for the slope of the linear fit of two variables.
    
    Arguments:
    x -- A 1D NumPy array or list representing the independent variable.
    y -- A 1D NumPy array or list representing the dependent variable.
    num_iterations -- The number of iterations to perform for the resampling (default: 1000).
    
    Returns:
    p_value -- The resampling p-value.
    """
    x = np.array(x)
    y = np.array(y)
    observed_slope = np.polyfit(x, y, 1)[0]
    num_data = len(x)
    larger_slope_count = 0

    for _ in tqdm.tqdm(range(num_iterations)):
        indices = np.random.choice(num_data, num_data, replace=True)
        resampled_x = x
        resampled_y = y[indices]
        resampled_slope = np.polyfit(resampled_x, resampled_y, 1)[0]
        if np.abs(resampled_slope) >= np.abs(observed_slope):
            larger_slope_count += 1

    p_value = (larger_slope_count + 1) / (num_iterations + 1)
    return p_value

def uniqlist(seq):
    #from https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


