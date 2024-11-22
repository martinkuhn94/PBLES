import os
import pickle

import pandas as pd
import tensorflow as tf
import yaml
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Masking,
    GRU,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    SimpleRNN,
)

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)

from PBLES.preprocessing.log_preprocessing import preprocess_event_log
from PBLES.preprocessing.log_tokenization import tokenize_log
from PBLES.sampling.log_sampling import sample_batch_new
from PBLES.postprocessing.log_postprocessing import generate_df


class EventLogDpLstm:
    def __init__(
            self,
            embedding_output_dims=16,
            method="LSTM",
            units_per_layer=None,
            epochs=3,
            batch_size=16,
            max_clusters=10,
            dropout=0.0,
            trace_quantile=0.95,
            l2_norm_clip=1.5,
            epsilon=1.0,
            num_attention_heads=4,  # Number of heads for multi-head attention
    ):
        # Initialization as before
        self.dict_dtypes = None
        self.cluster_dict = None
        self.event_log_sentences = None
        self.max_clusters = max_clusters
        self.trace_quantile = trace_quantile
        self.event_attribute_model = None

        self.model = None
        self.max_sequence_len = None
        self.total_words = None
        self.tokenizer = None
        self.ys = None
        self.xs = None
        self.start_epoch = None
        self.num_cols = None
        self.column_list = None

        self.units_per_layer = units_per_layer
        self.method = method
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads  # Storing the number of attention heads

        # Privacy Information
        self.noise_multiplier = None
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_examples = None

    def fit(self, input_data: pd.DataFrame) -> None:
        # Preprocessing as before
        (
            self.event_log_sentences,
            self.cluster_dict,
            self.dict_dtypes,
            self.start_epoch,
            self.num_examples,
            self.event_attribute_model,
            self.noise_multiplier,
            self.num_cols,
            self.column_list
        ) = preprocess_event_log(
            input_data, self.max_clusters, self.trace_quantile, (self.epsilon / 2), self.batch_size, self.epochs
        )

        (self.xs, self.ys, self.total_words, self.max_sequence_len, self.tokenizer) = tokenize_log(
            self.event_log_sentences, variant="attributes", steps=self.num_cols
        )

        # Define model architecture with flexible recurrent layer selection
        inputs = Input(shape=(self.max_sequence_len,))
        embedding_layer = Embedding(
            self.total_words, self.embedding_output_dims, input_length=self.max_sequence_len
        )(inputs)
        x = Masking(mask_value=0)(embedding_layer)

        # Recurrent or attention layers
        for i, units in enumerate(self.units_per_layer):
            if self.method == "LSTM":
                x = LSTM(units, return_sequences=True)(x)
            elif self.method == "Bi-LSTM":
                x = Bidirectional(LSTM(units, return_sequences=True))(x)
            elif self.method == "GRU":
                x = GRU(units, return_sequences=True)(x)
            elif self.method == "Attention-LSTM":
                x = LSTM(units, return_sequences=True)(x)  # LSTM layer
                x = MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=units)(x, x)  # Attention layer
            elif self.method == "RNN":
                x = SimpleRNN(units, return_sequences=True)(x)
            elif self.method == "Attention-RNN":
                x = SimpleRNN(units, return_sequences=True)(x)  # RNN layer
                x = MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=units)(x, x)  # Attention layer
            elif self.method == "Bi-RNN":
                x = Bidirectional(SimpleRNN(units, return_sequences=True))(x)
            elif self.method == "Bi-RNN-Attention":
                x = Bidirectional(SimpleRNN(units, return_sequences=True))(x)  # Bidirectional RNN layer
                x = MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=units)(x, x)  # Attention layer
            elif self.method == "Bi-GRU":
                x = Bidirectional(GRU(units, return_sequences=True))(x)
            elif self.method == "Bi-GRU-Attention":
                x = Bidirectional(GRU(units, return_sequences=True))(x)  # Bidirectional GRU layer
                x = MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=units)(x, x)  # Attention layer
            elif self.method == "Attention-GRU":
                x = GRU(units, return_sequences=True)(x)  # GRU layer
                x = MultiHeadAttention(num_heads=self.num_attention_heads, key_dim=units)(x, x)  # Attention layer

        # Batch Normalization and Dropout layers
        x = GlobalAveragePooling1D()(x)  # Summarize sequence information
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        # Dynamically create multiple output layers based on the number of steps
        outputs = []
        modified_column_list = []
        for column in self.column_list:
            modified_column_list.append(column.replace(":", "_").replace(" ", "_"))

        for step in range(self.num_cols):
            output = Dense(self.total_words, activation="softmax", name=f"{modified_column_list[step]}")(x)
            outputs.append(output)

        # Differentially Private Optimizer
        dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001,
        )

        # Compile the model with multiple outputs
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,  # Loss for each step
            optimizer=dp_optimizer,
            metrics=["accuracy"],
        )

        # Prepare multi-output labels
        y_outputs = [self.ys[:, step] for step in range(self.num_cols)]

        # Modify EarlyStopping to monitor accuracy for the first output
        early_stopping = EarlyStopping(
            monitor="output_step_1_accuracy",  # Monitor accuracy for the first output
            mode="max",  # Maximize accuracy
            verbose=1,
            patience=5
        )

        # Fit the model
        self.model.fit(
            self.xs,
            y_outputs,  # List of outputs for each step
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
        )

    def sample(self, sample_size: int, batch_size: int, temperature: float = 1.0) -> pd.DataFrame:
        """
        Sample an event log from a trained DP-Bi-LSTM Model. The model must be trained before sampling. The sampling
        process can be controlled by the temperature parameter, which controls the randomness of sampling process.
        A higher temperature results in more randomness.

        Parameters:
        sample_size (int): Number of traces to sample.
        temperature (float): Temperature for sampling the attribute information (default is 1.0).

        Returns:
        pd.DataFrame: DataFrame containing the sampled event log.
        """
        len_synthetic_event_log = 0
        synthetic_df = pd.DataFrame()

        while len_synthetic_event_log < sample_size:
            print("Sampling Event Log with:", sample_size - len_synthetic_event_log, "traces left")
            sample_size_new = sample_size - len_synthetic_event_log

            synthetic_event_log_sentences = sample_batch_new(
                sample_size_new,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                self.event_attribute_model,
                batch_size,
                temperature,
                self.num_cols,
                self.column_list
            )

            # Generate Event Log DataFrame
            df = generate_df(synthetic_event_log_sentences, self.cluster_dict, self.dict_dtypes, self.start_epoch)

            df.reset_index(drop=True, inplace=True)

            synthetic_df = pd.concat([synthetic_df, df], axis=0, ignore_index=True)
            len_synthetic_event_log += df["case:concept:name"].nunique()

        return synthetic_df

    def save_model(self, path: str) -> None:
        """
        Save a trained PBLES Model to a given path.

        Parameters:
        path (str): Path to save the trained PBLES Model.
        """
        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))

        self.event_attribute_model.to_excel(os.path.join(path, "event_attribute_model.xlsx"), index=False)

        with open(os.path.join(path, "tokenizer.pkl"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "cluster_dict.pkl"), "wb") as handle:
            pickle.dump(self.cluster_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "dict_dtypes.yaml"), "w", encoding='utf-8') as handle:
            yaml.dump(self.dict_dtypes, handle, default_flow_style=False)

        with open(os.path.join(path, "max_sequence_len.pkl"), "wb") as handle:
            pickle.dump(self.max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "start_epoch.pkl"), "wb") as handle:
            pickle.dump(self.start_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "num_cols.pkl"), "wb") as handle:
            pickle.dump(self.num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "column_list.pkl"), "wb") as handle:
            pickle.dump(self.column_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """
        Load a trained PBLES Model from a given path.

        Parameters:
        path (str): Path to the trained PBLES Model.
        """
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"), compile=False)

        self.event_attribute_model = pd.read_excel(os.path.join(path, "event_attribute_model.xlsx"))

        with open(os.path.join(path, "tokenizer.pkl"), "rb") as handle:
            self.tokenizer = pickle.load(handle)

        with open(os.path.join(path, "cluster_dict.pkl"), "rb") as handle:
            self.cluster_dict = pickle.load(handle)

        with open(os.path.join(path, "dict_dtypes.yaml"), "r", encoding='utf-8') as handle:
            self.dict_dtypes = yaml.safe_load(handle)

        with open(os.path.join(path, "max_sequence_len.pkl"), "rb") as handle:
            self.max_sequence_len = pickle.load(handle)

        with open(os.path.join(path, "start_epoch.pkl"), "rb") as handle:
            self.start_epoch = pickle.load(handle)

        with open(os.path.join(path, "num_cols.pkl"), "rb") as handle:
            self.num_cols = pickle.load(handle)

        with open(os.path.join(path, "column_list.pkl"), "rb") as handle:
            self.column_list = pickle.load(handle)
