import re
import os

import numpy as np
import pandas as pd
import pm4py
from sklearn.cluster import KMeans
from datetime import datetime
from diffprivlib.models import KMeans as DP_KMeans
from diffprivlib.mechanisms import Laplace
from tensorflow_privacy import compute_dp_sgd_privacy_statement

from PBLES.event_attribute_model.event_attribute_model import build_attribute_model
# TODO Konstante auslagern, conseptname, timestamp, case, start, end

os.environ["LOKY_MAX_CPU_COUNT"] = str(max(os.cpu_count() - 1, 1))

def extract_epsilon_from_string(text):
    """
    Poisson sampling is not usually done in training pipelines, but assuming
    that the data was randomly shuffled, it is believed that the actual epsilon
    should be closer to this value than the conservative assumption of an arbitrary
    data order.

    :param text:
    :return:
    """
    epsilon_poisson_match = re.search(r"Epsilon assuming Poisson sampling \(\*\):\s+([^\s]+)", text)

    if epsilon_poisson_match:
        epsilon_poisson = epsilon_poisson_match.group(1)
    else:
        epsilon_poisson = None

    return float(epsilon_poisson)


def find_noise_multiplier(target_epsilon, num_examples, batch_size, epochs, tol=1e-4, max_iter=100):
    delta = 1 / (num_examples ** 1.1)
    low, high = 1e-6, 30  # Initial bounds for noise multiplier
    best_noise_multiplier = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        current_epsilon = compute_dp_sgd_privacy_statement(
            number_of_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            noise_multiplier=mid,
            used_microbatching=False,
            delta=delta,
        )

        current_epsilon = extract_epsilon_from_string(current_epsilon)

        if abs(current_epsilon - target_epsilon) <= tol:
            best_noise_multiplier = mid
            break

        if current_epsilon > target_epsilon:
            low = mid  # Increase noise
        else:
            high = mid  # Decrease noise

    if best_noise_multiplier is None:
        best_noise_multiplier = high
        print(
            f"Warning: Noise multiplier could not be found within the maximum number of iterations. "
            f"Choosing the highest noise multiplier: {best_noise_multiplier}"
            f"Consider choosing another Epsilon values better suited to the dataset and the model configurations"
        )
    else:
        print(f"Optimal Noise multiplier found: {best_noise_multiplier}")
        print("Since three differential Privacy Techniques are used, the epsilon is divided in the following way: ")
        print(f"DP Bounds: {target_epsilon * 0.5}")
        print(f"DP-KMeans: {target_epsilon * 0.5}")
        print(f"DP-SDG: {target_epsilon}")

    return best_noise_multiplier


def calculate_dp_bounds(df, epsilon, std_multiplier=2):
    """
    Calculate differentially private bounds using mean ± (std_multiplier * std) approach.
    """
    dp_bounds = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) > 1:
            true_mean = float(col_data.mean())
            true_std = float(col_data.std())

            mean_sensitivity = true_std / np.sqrt(len(col_data))
            std_sensitivity = true_std / np.sqrt(2 * (len(col_data) - 1))

            laplace_mechanism_mean = Laplace(epsilon=epsilon / 2, sensitivity=mean_sensitivity)
            laplace_mechanism_std = Laplace(epsilon=epsilon / 2, sensitivity=std_sensitivity)

            dp_mean = laplace_mechanism_mean.randomise(true_mean)
            dp_std = abs(laplace_mechanism_std.randomise(true_std))
        else:
            # Handle the case where there's insufficient data
            dp_mean = np.nan
            dp_std = np.nan

        if col == "time:timestamp":
            min_value = 0
            noisy_max = dp_mean + (std_multiplier * dp_std)
            noisy_max = max(min_value + 1e-5, noisy_max)
            dp_bounds[col] = ([min_value], [noisy_max])
        else:
            noisy_lower = dp_mean - (std_multiplier * dp_std)
            noisy_upper = dp_mean + (std_multiplier * dp_std)
            dp_bounds[col] = ([noisy_lower], [noisy_upper])

    return dp_bounds

# TODO kleine funktionen
def calculate_cluster_dp(df, max_clusters, epsilon):
    """
    Calculate clusters for each numeric column in a pandas DataFrame using DP-KMeans and differentially private bounds.

    Parameters:
    df: Pandas DataFrame.
    max_clusters: Number of maximum clusters.
    epsilon: Privacy budget for DP-KMeans.

    Returns:
    tuple: A tuple containing a Pandas DataFrame with cluster labels and a dictionary with cluster information.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    epsilon_bounds = epsilon * 0.5
    epsilon_k_means = epsilon * 0.5

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_org = df.copy()
    df_cluster_list = []

    dp_bounds = calculate_dp_bounds(df, epsilon_bounds)

    for col in numeric_cols:
        df_clean = df[col].dropna()
        unique_values = len(df_clean.unique())
        if unique_values == 0:
            continue
        elif unique_values < max_clusters:
            n_clusters = unique_values
        else:
            n_clusters = max_clusters

        X = df_clean.values.reshape(-1, 1)

        bounds = dp_bounds[col]
        dp_kmeans = DP_KMeans(n_clusters=n_clusters, epsilon=epsilon_k_means, bounds=bounds, random_state=0)
        dp_kmeans.fit(X)

        label = []
        for row in df.iterrows():
            if str(row[1][col]) != "nan":
                label_temp = dp_kmeans.predict([[row[1][col]]])
                label.append(col + "_cluster_" + str(label_temp[0]))
            else:
                label.append(np.nan)
        df[col] = label
        df_org[col + "_cluster_label"] = label
        df_cluster_list.append(df_org[[col, col + "_cluster_label"]].dropna())

    cluster_dict = {}
    for dataframe in df_cluster_list:
        unique_cluster = dataframe[dataframe.columns[1]].unique()
        for cluster in unique_cluster:
            dataframe_temp_values = dataframe[dataframe[dataframe.columns[1]] == cluster]
            dataframe_temp_cluster_values = dataframe_temp_values[dataframe_temp_values.columns[0]]
            dataframe_temp_cluster_values_np = dataframe_temp_cluster_values.to_numpy()
            cluster_dict[cluster] = [
                min(dataframe_temp_cluster_values_np),
                max(dataframe_temp_cluster_values_np),
                dataframe_temp_cluster_values_np.mean(),
                dataframe_temp_cluster_values_np.std(),
            ]

    return df, cluster_dict

# TOOD Calculate_clsuter und dp sind fast identisch: https://dfkide-my.sharepoint.com/:i:/g/personal/jogr04_dfki_de/EVj6Stcyv-FCvn2lbeeiUOQBe-jGNjjxN1QAjwpu8_EsyQ?e=kaY8Ky die Codeduplikate sollten ausgelagert werden in eigene Funktionen
# evtl apply_clustering, calculate_clusters als ausgelagerte Funktionen
def calculate_cluster(df, max_clusters):
    """
    Calculate clusters for each numeric column in a pandas DataFrame using KMeans.

    Parameters:
    df: Pandas DataFrame.
    max_clusters: Number of maximum clusters.

    Returns:
    tuple: A tuple containing a Pandas DataFrame with cluster labels and a dictionary with cluster information.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame")

    if not isinstance(max_clusters, int) or max_clusters <= 0:
        raise ValueError("max_clusters must be a positive integer")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_org = df.copy()
    df_cluster_list = []

    for col in numeric_cols:
        df_clean = df[col].dropna()
        unique_values = len(df_clean.unique())
        if unique_values == 0:
            continue
        elif unique_values < max_clusters:
            n_clusters = unique_values
        else:
            n_clusters = max_clusters

        X = df_clean.values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(X)

        label = []
        for row in df.iterrows():
            if str(row[1][col]) != "nan":
                label_temp = kmeans.predict([[row[1][col]]])
                label.append(col + "_cluster_" + str(label_temp[0]))
            else:
                label.append(np.nan)
        df[col] = label
        df_org[col + "_cluster_label"] = label
        df_cluster_list.append(df_org[[col, col + "_cluster_label"]].dropna())

    cluster_dict = {}
    for dataframe in df_cluster_list:
        unique_cluster = dataframe[dataframe.columns[1]].unique()
        for cluster in unique_cluster:
            dataframe_temp_values = dataframe[dataframe[dataframe.columns[1]] == cluster]
            dataframe_temp_cluster_values = dataframe_temp_values[dataframe_temp_values.columns[0]]
            dataframe_temp_cluster_values_np = dataframe_temp_cluster_values.to_numpy()
            cluster_dict[cluster] = [
                min(dataframe_temp_cluster_values_np),
                max(dataframe_temp_cluster_values_np),
                dataframe_temp_cluster_values_np.mean(),
                dataframe_temp_cluster_values_np.std(),
            ]

    return df, cluster_dict


def calculate_starting_epoch_dp(df: pd.DataFrame, epsilon: float) -> list:
    """
    Calculate differentially private starting epoch statistics for an event log.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                       'case:concept:name' and 'time:timestamp'.
    epsilon (float): Privacy budget for differential privacy.

    Returns:
    list: A list containing four elements:
          [DP Mean, DP Standard Deviation, DP Min (fixed at 0), Max (current timestamp)].

    Raises:
    ValueError: If required columns are missing or if there are issues in date conversion.
    """
    try:
        if "case:concept:name" not in df or "time:timestamp" not in df:
            raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        starting_epochs = df.sort_values(by="time:timestamp").groupby("case:concept:name")["time:timestamp"].first()

        # Convert timestamps to UNIX time (seconds since epoch)
        starting_epoch_list = starting_epochs.astype(np.int64) // 10 ** 9

        if len(starting_epoch_list) == 0:
            raise ValueError("No valid starting timestamps found in the data.")

        starting_epoch_mean = np.mean(starting_epoch_list)
        starting_epoch_std = np.std(starting_epoch_list)

        starting_epoch_min = 0
        max_timestamp = int(datetime.now().timestamp())
        n_traces = len(starting_epoch_list)
        range_epochs = max_timestamp - starting_epoch_min

        sensitivity_mean = range_epochs / n_traces
        sensitivity_std = range_epochs / np.sqrt(2 * n_traces)

        laplace_mechanism_mean = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity_mean)
        dp_mean = abs(laplace_mechanism_mean.randomise(starting_epoch_mean))

        laplace_mechanism_std = Laplace(epsilon=epsilon / 2, sensitivity=sensitivity_std)
        dp_std = abs(laplace_mechanism_std.randomise(starting_epoch_std))

        return [dp_mean, dp_std, starting_epoch_min, max_timestamp]

    except Exception as e:
        raise ValueError(f"An error occurred in calculating differentially private starting epochs: {str(e)}")


def calculate_starting_epoch(df: pd.DataFrame) -> list:
    """
    Calculate starting epoch statistics for an event log.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                      'case:concept:name' and 'time:timestamp'.

    Returns:
    list: A list containing four elements:
          [Mean, Standard Deviation, Min (fixed at 0), Max (current timestamp)].

    Raises:
    ValueError: If required columns are missing or if there are issues in date conversion.
    """
    try:
        if "case:concept:name" not in df or "time:timestamp" not in df:
            raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        starting_epochs = df.sort_values(by="time:timestamp").groupby("case:concept:name")["time:timestamp"].first()

        # Convert timestamps to UNIX time (seconds since epoch)
        starting_epoch_list = starting_epochs.astype(np.int64) // 10 ** 9

        if len(starting_epoch_list) == 0:
            raise ValueError("No valid starting timestamps found in the data.")

        starting_epoch_mean = np.mean(starting_epoch_list)
        starting_epoch_std = np.std(starting_epoch_list)
        starting_epoch_min = 0
        max_timestamp = int(datetime.now().timestamp())

        return [starting_epoch_mean, starting_epoch_std, starting_epoch_min, max_timestamp]

    except Exception as e:
        raise ValueError(f"An error occurred in calculating starting epochs: {str(e)}")


def calculate_time_between_events(df: pd.DataFrame) -> list:
    """
    Calculate the time between events for each trace in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame representing an event log, expected to contain columns
                       'case:concept:name' and 'time:timestamp'.

    Returns:
    list: A list of time between events for each trace in the DataFrame, given in seconds as Unix time.
    """
    if "case:concept:name" not in df or "time:timestamp" not in df:
        raise ValueError("DataFrame must contain 'case:concept:name' and 'time:timestamp' columns")

    try:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    except Exception as e:
        raise ValueError(f"Error converting 'time:timestamp' to datetime: {e}")

    time_between_events = []

    for _, group in df.groupby("case:concept:name"):
        if len(group) < 2:
            time_between_events.append(0)
            continue

        time_diffs = group["time:timestamp"].diff().dt.total_seconds().copy()
        time_diffs.fillna(0, inplace=True)
        time_diffs.iloc[0] = 0
        time_between_events.extend(time_diffs)

    return time_between_events


def get_attribute_dtype_mapping(df: pd.DataFrame) -> dict:
    """
    Get the attribute data type mapping from an Event Log (XES) and return it as dictionary.
    This is necessary to generate synthetic data, maintaining the correct datatypes from the original data.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame representing an event log, where columns are attributes.

    Returns:
    dict: Dictionary containing the attribute data type mapping
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    dtype_dict = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if column == 'time:timestamp':
                dtype_dict[column] = 'float64'
            elif df[column].dropna().apply(lambda x: float(x).is_integer()).all():
                dtype_dict[column] = 'int64'
            else:
                dtype_dict[column] = 'float64'
        else:
            dtype_dict[column] = df[column].dtype.name

    return {'attribute_datatypes': dtype_dict}

# TODO in unterfunktionen aufteilen
def preprocess_event_log(log, max_clusters: int, trace_quantile: float, epsilon: float, batch_size: int, epochs: int):
    try:
        df = pm4py.convert_to_dataframe(log)
    except Exception as e:
        raise ValueError(f"Error converting log to DataFrame: {e}")

    print("Number of traces: " + str(df["case:concept:name"].unique().size))

    trace_length = df.groupby("case:concept:name").size()
    trace_length_q = trace_length.quantile(trace_quantile)
    df = df.groupby("case:concept:name").filter(lambda x: len(x) <= trace_length_q)

    print("Number of traces after truncation: " + str(df["case:concept:name"].unique().size))
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    num_examples = len(df)

    if epsilon is None:
        print("No Epsilon is specified setting noise multiplier to 0")
        noise_multiplier = 0
        starting_epoch_dist = calculate_starting_epoch(df)
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_cluster(df, max_clusters)
    else:
        print("Finding Optimal Noise Multiplier")
        epsilon_noise_multiplier = epsilon / 2
        epsilon_k_means = epsilon / 2
        noise_multiplier = find_noise_multiplier(epsilon_noise_multiplier, num_examples, batch_size, epochs)
        starting_epoch_dist = calculate_starting_epoch_dp(df, epsilon)  # Epsilon does not need to be shared here since the first timestamp defines a distinct dataset.
        time_between_events = calculate_time_between_events(df)
        df["time:timestamp"] = time_between_events
        attribute_dtype_mapping = get_attribute_dtype_mapping(df)
        df, cluster_dict = calculate_cluster_dp(df, max_clusters, epsilon_k_means)

    cols = ["concept:name", "time:timestamp"] + [
        col for col in df.columns if col not in ["concept:name", "time:timestamp"]
    ]
    df = df[cols]
    event_attribute_model = build_attribute_model(df)

    event_log_sentence_list = []
    total_traces = df["case:concept:name"].nunique()

    num_cols = len(df.columns) - 1
    column_list = df.columns.tolist()

    if 'case:concept:name' in column_list:
        column_list.remove('case:concept:name')

    for i, trace in enumerate(df["case:concept:name"].unique(), start=1):
        print(f"\rProcessing trace {i} of {total_traces}", end="", flush=True)
        df_temp = df[df["case:concept:name"] == trace]
        df_temp = df_temp.drop(columns=['case:concept:name'])
        trace_sentence_list = ["START==START"] * num_cols

        for global_attribute in df_temp:
            if global_attribute.startswith("case:") and global_attribute != "case:concept:name":
                trace_sentence_list.append(global_attribute + "==" + str(df_temp[global_attribute].iloc[0]))

        for row in df_temp.iterrows():
            concept_name = row[1]["concept:name"]
            for col in df_temp.columns:
                trace_sentence_list.append(concept_name + "==" + col + "==" + str(row[1][col]) if str(
                    row[1][col]) != "nan" else concept_name + "==" + col + "==" + "nan")

        trace_sentence_list.extend(["END==concept:name==END"] * num_cols)
        event_log_sentence_list.append(trace_sentence_list)

    print()
    return (
        event_log_sentence_list,
        cluster_dict,
        attribute_dtype_mapping,
        starting_epoch_dist,
        num_examples,
        event_attribute_model,
        noise_multiplier,
        num_cols,
        column_list
    )
