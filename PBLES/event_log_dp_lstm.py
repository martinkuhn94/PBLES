import os
import pickle
import time

import pandas as pd
import tensorflow as tf
import yaml
from keras import Input, Model
from tensorflow.keras.callbacks import Callback
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
    GlobalAveragePooling1D,
    SimpleRNN,
)

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasAdamOptimizer,
)

from PBLES.preprocessing.log_preprocessing import preprocess_event_log
from PBLES.preprocessing.log_tokenization import tokenize_log
from PBLES.sampling.log_sampling import sample_batch
from PBLES.postprocessing.log_postprocessing import generate_df


class MetricsLogger(Callback):
    def __init__(self, num_cols, column_list):
        super().__init__()
        self.num_cols = num_cols
        self.column_list = [col.replace(":", "_").replace(" ", "_") for col in column_list]
        self.history = []
        # Disable logging for this callback
        self._supports_tf_logs = False

    def on_epoch_end(self, epoch, logs=None):
        epoch_metrics = {'epoch': epoch + 1}
        logs = logs or {}

        # Silently collect metrics
        for i in range(self.num_cols):
            output_acc = f'{self.column_list[i]}_accuracy'
            output_loss = f'{self.column_list[i]}_loss'

            if output_acc in logs:
                epoch_metrics[output_acc] = logs[output_acc]
            if output_loss in logs:
                epoch_metrics[output_loss] = logs[output_loss]

        if 'loss' in logs:
            epoch_metrics['total_loss'] = logs['loss']

        self.history.append(epoch_metrics)

    def get_dataframe(self):
        return pd.DataFrame(self.history)


# Custom callback to handle progress bar only
class CustomProgressBar(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.last_update = None
        self.start_time = None
        self.target = None
        self.seen = None

    def on_epoch_begin(self, epoch, logs=None):
        print(f'\nEpoch {epoch + 1}/{self.params["epochs"]}')
        self.seen = 0
        self.target = self.params['steps']
        self.start_time = time.time()
        self.last_update = time.time()

    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        now = time.time()

        # Calculate time per step and ETA
        time_elapsed = now - self.start_time
        steps_remaining = self.target - self.seen
        time_per_step = time_elapsed / self.seen
        eta_seconds = steps_remaining * time_per_step

        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"

        # Calculate progress bar
        progress = int(30 * self.seen / self.target)
        bar = '=' * progress + '>' + '.' * (29 - progress)

        # Calculate time per step (ms)
        time_per_step_ms = time_per_step * 1000

        print(f'\r{self.seen}/{self.target} [{bar}] - ETA: {eta_str} - {time_per_step_ms:.0f}ms/step', end='')

    def on_epoch_end(self, epoch, logs=None):
        total_time = time.time() - self.start_time
        if total_time < 60:
            time_str = f"{total_time:.0f}s"
        elif total_time < 3600:
            time_str = f"{int(total_time / 60)}m {int(total_time % 60)}s"
        else:
            time_str = f"{int(total_time / 3600)}h {int((total_time % 3600) / 60)}m"

        print(f'\r{self.target}/{self.target} [==============================] - {time_str}')


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
            epsilon=None,
            num_attention_heads=4,  # Number of heads for multi-head attention
    ):
        # Initialization as before
        self.modified_column_list = None
        self.metrics_df = None
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

    # Split the fit function into two parts: initialize_model and train

    def initialize_model(self, input_data: pd.DataFrame) -> None:
        # Do the preprocessing and model setup only once
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
            input_data, self.max_clusters, self.trace_quantile, self.epsilon, self.batch_size, self.epochs
        )

        (self.xs, self.ys, self.total_words, self.max_sequence_len, self.tokenizer) = tokenize_log(
            self.event_log_sentences, variant="attributes", steps=self.num_cols
        )

        # Build model architecture
        inputs = Input(shape=(self.max_sequence_len,))
        embedding_layer = Embedding(
            self.total_words, self.embedding_output_dims, input_length=self.max_sequence_len
        )(inputs)
        x = Masking(mask_value=0)(embedding_layer)

        """
        Es wäre auch als dictionary factory möglich

        layer_factory = {
        "LSTM": lambda units: LSTM(units, return_sequences=True),
        "Bi-LSTM": lambda units: Bidirectional(LSTM(units, return_sequences=True)),
        "GRU": lambda units: GRU(units, return_sequences=True),
        "Bi-GRU": lambda units: Bidirectional(GRU(units, return_sequences=True)),
        "RNN": lambda units: SimpleRNN(units, return_sequences=True),
        "Bi-RNN": lambda units: Bidirectional(SimpleRNN(units, return_sequences=True)),
        }
        
        # Build the layers based on the method and units per layer
        for i, units in enumerate(self.units_per_layer):
            if self.method not in layer_factory:
                raise ValueError(f"Unsupported method: {self.method}")
            if not isinstance(self.units_per_layer, list):
                raise ValueError("`units_per_layer` should be a list of integers.")
            x = layer_factory[self.method](units)(x)
        """
        
        for i, units in enumerate(self.units_per_layer):
            if self.method == "LSTM":
                x = LSTM(units, return_sequences=True)(x)
            elif self.method == "Bi-LSTM":
                x = Bidirectional(LSTM(units, return_sequences=True))(x)
            elif self.method == "GRU":
                x = GRU(units, return_sequences=True)(x)
            elif self.method == "Bi-GRU":
                x = Bidirectional(GRU(units, return_sequences=True))(x)
            elif self.method == "RNN":
                x = SimpleRNN(units, return_sequences=True)(x)
            elif self.method == "Bi-RNN":
                x = Bidirectional(SimpleRNN(units, return_sequences=True))(x)

        x = GlobalAveragePooling1D()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        outputs = []
        self.modified_column_list = []
        for column in self.column_list:
            self.modified_column_list.append(column.replace(":", "_").replace(" ", "_"))

        for step in range(self.num_cols):
            output = Dense(self.total_words, activation="softmax", name=f"{self.modified_column_list[step]}")(x)
            outputs.append(output)

        # Differentially Private Optimizer
        dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001,
        )

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=["sparse_categorical_crossentropy"] * self.num_cols,
            optimizer=dp_optimizer,
            metrics=["accuracy"],
        )

    def train(self, epochs: int) -> None:
        # Training function that can be called multiple times
        y_outputs = [self.ys[:, step] for step in range(self.num_cols)]

        early_stopping = EarlyStopping(
            monitor=f"{self.modified_column_list[0]}_accuracy",
            mode="max",
            verbose=0,
            patience=7,
            restore_best_weights=True,
            min_delta=0.001,
            baseline=None,
            start_from_epoch=5
        )

        metrics_logger = MetricsLogger(num_cols=self.num_cols, column_list=self.column_list)
        custom_progress_bar = CustomProgressBar()

        self.model.fit(
            self.xs,
            y_outputs,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, metrics_logger, custom_progress_bar],
            verbose=0
        )

        self.metrics_df = metrics_logger.get_dataframe()

    def fit(self, input_data: pd.DataFrame) -> None:
        # Keep original fit function for backward compatibility
        self.initialize_model(input_data)
        self.train(self.epochs)
        
    def sample(self, sample_size: int, batch_size: int) -> pd.DataFrame:
        """
        Sample an event log from a trained DP-Bi-LSTM Model. The model must be trained before sampling. The sampling
        process can be controlled by the temperature parameter, which controls the randomness of sampling process.
        A higher temperature results in more randomness.

        Parameters:
        sample_size (int): Number of traces to sample.
        batch_size (int): Number of traces to sample in a batch.

        Returns:
        pd.DataFrame: DataFrame containing the sampled event log.
        """
        len_synthetic_event_log = 0
        synthetic_df = pd.DataFrame()

        while len_synthetic_event_log < sample_size:
            # TODO: ich würde alle prints durch Logging ersetzen
            print("Sampling Event Log with:", sample_size - len_synthetic_event_log, "traces left")
            sample_size_new = sample_size - len_synthetic_event_log

            synthetic_event_log_sentences = sample_batch(
                sample_size_new,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                batch_size,
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

"""
TODO
wäre es evtl. gut die Hyperparameter auch abzuspeichern?
model:
  embedding_output_dims: 16
  method: LSTM
  units_per_layer: [64, 32]
  epochs: 10
  batch_size: 32
privacy:
  l2_norm_clip: 1.5
  epsilon: 1.0

  oder ist das unten das yaml dump? 
  """

        
        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))

        self.event_attribute_model.to_excel(os.path.join(path, "event_attribute_model.xlsx"), index=False)

        self.metrics_df.to_excel(os.path.join(path, "training_metrics.xlsx"), index=False)

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
