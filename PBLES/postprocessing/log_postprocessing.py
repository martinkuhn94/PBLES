import datetime
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


def generate_df(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch) -> pd.DataFrame:
    """
    Generate a DataFrame from synthetic event log sentences.

    Parameters:
    synthetic_event_log_sentences: List of synthetic event log sentences.
    cluster_dict: Dictionary of cluster information.
    dict_dtypes: Dictionary of data types.
    start_epoch: List containing start epoch information.
    event_attribute_dict: Dictionary containing event attribute mappings.

    Returns:
    pd.DataFrame: Generated DataFrame.
    """
    print("Creating DF-Event Log from synthetic Data")
    transformed_sentences = transform_sentences(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch)

    # Create DataFrame from transformed sentences
    df = create_dataframe_from_sentences(transformed_sentences, dict_dtypes)

    # Clean and reorder DataFrame
    df = reorder_and_sort_df(df)

    return df


def reorder_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by 'case:concept:name' and 'time:timestamp' if these columns are present.
    Make 'case:concept:name' the first column, 'concept:name' the second column, and 'time:timestamp' the third.

    Parameters:
    df (pd.DataFrame): Input DataFrame to be reordered and sorted.

    Returns:
    pd.DataFrame: Reordered and sorted DataFrame.
    """
    if "case:concept:name" in df.columns and "time:timestamp" in df.columns:
        df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)

    columns_order = []
    if "case:concept:name" in df.columns:
        columns_order.append("case:concept:name")
    if "concept:name" in df.columns:
        columns_order.append("concept:name")
    if "time:timestamp" in df.columns:
        columns_order.append("time:timestamp")

    other_columns = [col for col in df.columns if col not in columns_order]
    df = df[columns_order + other_columns]

    return df


def create_start_epoch(start_epoch: list[float]) -> datetime.datetime:
    """
    Create a start epoch for the synthetic event log generation. The start epoch is generated using a normal
    distribution with the mean and standard deviation specified in the start_epoch list. The generated epoch is
    then checked against the min and max bounds. If the value is out of bounds, it is regenerated.

    Parameters:
    start_epoch (list[float]): List containing:
                               [mean, standard deviation, min bound, max bound]
                               for the normal distribution used to generate the start epoch.

    Returns:
    datetime.datetime: Start epoch as a datetime object.
    """
    dp_mean, dp_std, min_bound, max_bound = start_epoch  # Unpack the list
    epoch_dist = norm(loc=dp_mean, scale=dp_std)

    while True:
        # Generate a value from the normal distribution
        epoch_value = epoch_dist.rvs(1)[0]

        # Check if the value is within bounds
        if min_bound <= epoch_value <= max_bound:
            break

    # Convert to datetime and return
    epoch = datetime.datetime.fromtimestamp(epoch_value)
    return epoch


def transform_sentences(synthetic_event_log_sentences, cluster_dict, dict_dtypes, start_epoch) -> list[list[str]]:
    """
    Transform synthetic event log sentences by processing each word in the sentence and updating the temporary sentence.

    Parameters:
    synthetic_event_log_sentences: List of synthetic event log sentences.
    cluster_dict: Dictionary of cluster information.
    dict_dtypes: Dictionary of data types.
    start_epoch: List containing start epoch information.

    Returns:
    list: List of transformed synthetic event log sentences.
    """
    transformed_sentences = []
    for sentence, case_id in zip(synthetic_event_log_sentences, range(len(synthetic_event_log_sentences))):
        print(
            "\r"
            + "Converting into Event Log "
            + str(round((case_id + 1) / len(synthetic_event_log_sentences) * 100, 1))
            + "% Complete",
            end="",
        )
        sys.stdout.flush()

        temp_sentence = ["case:concept:name==" + str(datetime.datetime.now().timestamp()).replace(".", "")]
        epoch = create_start_epoch(start_epoch)
        for word in sentence:
            temp_sentence, epoch = process_word(word, temp_sentence, dict_dtypes, cluster_dict, epoch)

        transformed_sentences.append(temp_sentence)

    print("\n")

    return transformed_sentences


def process_word(word, temp_sentence, dict_dtypes, cluster_dict, epoch):
    """
    Process a word in the sentence and update the temporary sentence list.

    Parameters:
    word: The word to process.
    temp_sentence: The temporary sentence list to update.
    dict_dtypes: Dictionary of data types from YAML (nested under 'attribute_datatypes').
    cluster_dict: Dictionary of cluster information.
    epoch: Current epoch time.

    Returns:
    list: Updated temporary sentence list.
    """
    parts = word.split("==")
    if len(parts) == 2:
        key, value = parts
    else:
        key = parts[0]
        value = "0"

    dtype_mapping = dict_dtypes['attribute_datatypes']
    if key in dtype_mapping and key != "time:timestamp":
        if value in cluster_dict:
            generation_input = cluster_dict[value]
            dist = norm(loc=generation_input[2], scale=generation_input[3])
            value = dist.rvs(1)[0]
            value = round(value, 5) if dtype_mapping[key] in ["float", "float64"] else round(value)
            temp_sentence.append(f"{key}=={value}")
        else:
            temp_sentence.append(word)
    elif key == "time:timestamp":
        generation_input = cluster_dict[value]
        dist = norm(loc=generation_input[2], scale=generation_input[3])
        value = abs(dist.rvs(1)[0])
        epoch = epoch + datetime.timedelta(seconds=value)
        timestamp_string = epoch.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        if timestamp_string == "NaT":
            print("NaT was generated using previous Timestamp")
            timestamp_string = temp_sentence[-1].split("==")[1]
            timestamp_string = datetime.datetime.strptime(timestamp_string, "%Y-%m-%dT%H:%M:%S.%f+00:00")
            timestamp_string = timestamp_string + datetime.timedelta(seconds=1)
        temp_sentence.append(f"time:timestamp=={timestamp_string}")

    return temp_sentence, epoch


def create_dataframe_from_sentences(transformed_sentences, dict_dtypes) -> pd.DataFrame:
    """
    Create a DataFrame from the transformed synthetic event log sentences. The DataFrame is created by parsing the
    transformed sentences and converting the data types of the columns based on the dict_dtypes dictionary. The
    DataFrame is then sorted by 'case:concept:name' and 'time:timestamp' and the timestamps are interpolated and
    forward-filled.

    Parameters:
    transformed_sentences: List of transformed synthetic event log sentences.
    dict_dtypes: Dictionary of data types from YAML (nested under 'attribute_datatypes').

    Returns:
    pd.DataFrame: DataFrame created from the transformed synthetic event log sentences.
    """
    parsed_data = []
    removed_traces = 0

    for sentence in transformed_sentences:
        try:
            case_dict = {
                word.split("==")[0]: word.split("==")[1] for word in sentence if word.split("==")[0].startswith("case:")
            }
            event_indices = [i for i, s in enumerate(sentence) if s.startswith("concept:name")]
            event_indices.pop(0)
            events = np.split(sentence, event_indices)
            event_dict_list = []
            for event in events:
                event_dict = {word.split("==")[0]: word.split("==")[1] for word in event}
                event_dict.update(case_dict)
                event_dict_list.append(event_dict)
            parsed_data.append(event_dict_list)
        except Exception:
            removed_traces += 1

    df = pd.DataFrame()
    for case in parsed_data:
        df = pd.concat([df, pd.DataFrame(case)], ignore_index=True)

    # Access the nested dictionary structure
    dtype_mapping = dict_dtypes['attribute_datatypes']

    for key, value in dtype_mapping.items():
        if key in df.columns:
            df[key] = convert_column_dtype(df[key], value)

    if "time:timestamp" not in df.columns:
        df["time:timestamp"] = [pd.Timestamp("2000-01-01").strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")] * len(df)
    else:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.interpolate(method="ffill")
    )
    df["time:timestamp"] = df.groupby("case:concept:name")["time:timestamp"].transform(
        lambda x: x.ffill() if pd.isna(x.iloc[0]) else x
    )

    # Replace all nan values with empty string
    df = df.fillna("")
    # replace "nan" with ""
    df = df.replace("nan", "")

    return df


def convert_column_dtype(column: pd.Series, dtype: str) -> pd.Series:
    """
    Convert the data type of a column in the DataFrame based on the specified dtype. The conversion is done using the
    astype method of the pandas Series.

    Parameters:
    column: The column to convert.
    dtype: The data type to convert the column to.

    Returns:
    pd.Series: Converted column.
    """
    if dtype == "int":
        # round each element in column on next integer
        column = column.round()
        return column.astype(int)
    elif dtype == "float" or dtype == "float64":
        return column.astype(float) if column.name != "time:timestamp" else column.astype(str)
    elif dtype == "boolean":
        return column.astype(bool)
    elif dtype == "date":
        return column.astype(str)
    elif dtype == "string" or dtype == "object":
        return column.astype(str)

    return column
