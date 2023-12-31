import numpy as np

from keras import layers
from keras import losses
from keras import callbacks
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras import Input, Model
from project_functions.sample_feed_v1_multi import SampleFeed

import keras_tuner
import logging

logging.basicConfig(level=logging.INFO)

class LSTMHypermodel(keras_tuner.HyperModel):

    def __init__(
            self, 
            training_window_size, 
            predicted_window_size,
            n_features_seq,
            n_features_stc,
            samples_per_epoch,
            batch_size
            ):
        
        super().__init__()

        self.training_window_size = training_window_size
        self.predicted_window_size = predicted_window_size
        self.n_features_seq = n_features_seq
        self.n_features_stc = n_features_stc
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size

    def set_hp(self, hp: keras_tuner.HyperParameters):
        """Configure hyperparameters to tune."""

        hp.Float(name='LSTM_1_L2_recurrent', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
        hp.Float(name='LSTM_1_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.0)

        hp.Float(name='LSTM_2_L2_recurrent', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
        hp.Float(name='LSTM_2_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.0)
                
        hp.Float(name='DenseStatic_1_L2_kernel', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
        hp.Float(name='DenseStatic_1_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.1)

        hp.Float(name='Dense_1_L2_kernel', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
        hp.Float(name='Dense_1_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.1)

    def build(self, hp: keras_tuner.HyperParameters):

        self.set_hp(hp)

        input_seq  = Input(shape=(self.training_window_size, self.n_features_seq), name="TimeSeqInput")
        x_seq      = layers.LSTM(
                        units=32, 
                        return_sequences=True, 
                        recurrent_regularizer=regularizers.L2(hp.get('LSTM_1_L2_recurrent')), 
                        recurrent_dropout = hp.get('LSTM_1_dropout'),
                        name="LSTM_1"
                        )(input_seq)
        x_seq      = layers.LSTM(
                        units=16, 
                        return_sequences=False, 
                        recurrent_regularizer=regularizers.L2(hp.get('LSTM_1_L2_recurrent')), 
                        recurrent_dropout = hp.get('LSTM_2_dropout'),
                        name="LSTM_2"
                        )(x_seq)

        model_seq  = Model(inputs=input_seq, outputs=x_seq, name="TimeSeqModel")

        input_stc  = layers.Input(shape=(self.n_features_stc,), name="StaticInput")
        x_stc      = layers.Dense(
                        units=16, 
                        activation="relu",
                        kernel_regularizer=regularizers.L2(hp.get('DenseStatic_1_L2_kernel')),
                        name="DenseStatic_1"
                        )(input_stc)
        x_stc      = layers.Dropout(hp.get('DenseStatic_1_dropout'))(x_stc)
        model_stc  = Model(inputs=input_stc, outputs=x_stc, name="StaticModel")

        x_comb     = layers.concatenate([model_seq.output, model_stc.output], name="Concat")
        x_comb     = layers.Dense(
                        units=32, 
                        activation="relu", 
                        kernel_regularizer=regularizers.L2(hp.get('Dense_1_L2_kernel')),
                        name="Dense_1"
                        )(x_comb)
        x_comb     = layers.Dropout(hp.get('Dense_1_dropout'))(x_comb)
        x_comb     = layers.Dense(units=16, activation="relu", name="Dense_2")(x_comb)
        x_comb     = layers.Dense(units=self.predicted_window_size, activation="sigmoid", name="Output")(x_comb)

        model_comb = Model(inputs=[model_seq.input, model_stc.input], outputs=x_comb)
                
        model_comb.compile(
            loss=losses.Huber(0.25), 
            optimizer=optimizers.Adam(learning_rate=1e-3), 
            metrics=metrics.RootMeanSquaredError()
            )

        return model_comb
    
    def fit(self, hp: keras_tuner.HyperParameters, model, features_train, features_valid, epochs, *args, **kwargs):

        sample_feed = SampleFeed(self.training_window_size, self.predicted_window_size, self.samples_per_epoch)

        Xy_train_gen = sample_feed.random_sample_stream(features_train)
        Xy_valid = sample_feed.random_sample_array(features_valid, samples_per_page=1, shuffle=False, seed=0)
        
        n_rows = features_train['visits'].shape[0]
        self.steps_per_epoch = round(n_rows * self.samples_per_epoch / self.batch_size)

        return model.fit(
            x = Xy_train_gen,
            validation_data = Xy_valid,
            steps_per_epoch = self.steps_per_epoch,
            epochs = epochs,
            batch_size = self.batch_size
            *args,
            **kwargs
        )

TRAINING_WINDOW_SIZE = 90
PREDICTED_WINDOW_SIZE = 7

N_FEATURES_SEQ = 3
N_FEATURES_STC = 30

N_SAMPLES = 4
BATCH_SIZE = 8
N_EPOCHS = 12

MAX_TRIALS = 15
REDUCE_PAGES = 5
PROJECT_NAME = "try1230"

def run():
    """Run hyperparameter tuning"""

    features_train = dict(np.load("data/features_train.npz", allow_pickle=True))
    features_valid = dict(np.load("data/features_valid.npz", allow_pickle=True))

    features_train['visits'] = features_train['visits'][::REDUCE_PAGES, :]
    features_train['page'] = features_train['page'][::REDUCE_PAGES, :]

    features_valid['visits'] = features_valid['visits'][::REDUCE_PAGES, :]
    features_valid['page'] = features_valid['page'][::REDUCE_PAGES, :]

    lstm_hypermodel = LSTMHypermodel(
        training_window_size = TRAINING_WINDOW_SIZE,
        predicted_window_size = PREDICTED_WINDOW_SIZE,
        n_features_seq=N_FEATURES_SEQ,
        n_features_stc=N_FEATURES_STC,
        samples_per_epoch = N_SAMPLES,
        batch_size = BATCH_SIZE
    )

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=lstm_hypermodel,
        objective=keras_tuner.Objective(name='val_root_mean_squared_error', direction='min'),
        max_trials=MAX_TRIALS,
        directory='tuning',
        project_name=PROJECT_NAME
    )

    model_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]

    tuner.search_space_summary()

    tuner.search(
        features_train = features_train, 
        features_valid = features_valid,
        epochs = N_EPOCHS,
        callbacks = model_callbacks
        )
    
if __name__ == "__main__":
    run()