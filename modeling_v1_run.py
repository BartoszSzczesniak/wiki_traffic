import pickle
import numpy as np
from datetime import datetime
from project_functions.sample_feed_v1_multi import SampleFeed

from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras import Input, Model

TRAINING_WINDOW_SIZE = 90
PREDICTED_WINDOW_SIZE = 7

N_FEATURES_SEQ = 3
N_FEATURES_STC = 30

N_SAMPLES = 4
N_EPOCHS = 50
BATCH_SIZE = 8

def run():

    today_label = datetime.today().strftime("%m%d")

    # Raw data

    features_train = dict(np.load("data/features_train.npz", allow_pickle=True))
    features_valid = dict(np.load("data/features_valid.npz", allow_pickle=True))

    # Calculated parameters

    n_rows_train = features_train['visits'].shape[0]

    steps_per_epoch = round(n_rows_train * N_SAMPLES / BATCH_SIZE)
    total_samples_per_page = N_SAMPLES * N_EPOCHS

    sample_feed = SampleFeed(
        training_window_size = TRAINING_WINDOW_SIZE,
        predicted_window_size = PREDICTED_WINDOW_SIZE,
        samples_per_epoch = N_SAMPLES
        )

    Xy_train_gen = sample_feed.random_sample_stream(features_train)
    Xy_valid = sample_feed.random_sample_array(features_valid, samples_per_page=1, shuffle=False, seed=0)

    input_seq   = Input(shape=(TRAINING_WINDOW_SIZE, N_FEATURES_SEQ), name="TimeSeqInput")
    x_seq       = layers.LSTM(
                    units=32, 
                    # recurrent_dropout=0.2, 
                    return_sequences=True, 
                    name="LSTM_1"
                    )(input_seq)
    x_seq       = layers.LSTM(
                    units=16,
                    # recurrent_dropout=0.1, 
                    # recurrent_regularizer=regularizers.L2(0.01), 
                    name="LSTM_2"
                    )(x_seq)
    model_seq   = Model(inputs=input_seq, outputs=x_seq, name="TimeSeqModel")

    input_stc   = layers.Input(shape=(N_FEATURES_STC,), name="StaticInput")
    x_stc       = layers.Dense(
                    units=16,
                    # kernel_regularizer=regularizers.L2(0.01),
                    activation="relu", 
                    name="DenseStatic_1"
                    )(input_stc)
    # x_stc       = layers.Dropout(rate=0.1)(x_stc)
    model_stc   = Model(inputs=input_stc, outputs=x_stc, name="StaticModel")

    x_comb      = layers.concatenate([model_seq.output, model_stc.output], name="Concat")
    x_comb      = layers.Dense(
                    units=32, 
                    activation="relu", 
                    name="Dense_1"
                    )(x_comb)
    x_comb      = layers.Dense(
                    units=16, 
                    activation="relu", 
                    name="Dense_2"
                    )(x_comb)
    x_comb      = layers.Dense(
                    units=PREDICTED_WINDOW_SIZE, 
                    activation="sigmoid", 
                    name="Output"
                    )(x_comb)
    model_comb  = Model(inputs=[model_seq.input, model_stc.input], outputs=x_comb)

    model_comb.compile(
        loss=losses.Huber(0.25), 
        optimizer=optimizers.Adam(learning_rate=1e-3), 
        metrics=metrics.RootMeanSquaredError()
        )

    model_comb.summary()

    model_callbacks = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5),
        callbacks.EarlyStopping(monitor='val_loss', patience=8),
        callbacks.ModelCheckpoint(filepath=f"models/checkpoints/comb_{today_label}_" + "{epoch:02d}-{val_root_mean_squared_error:.6f}.keras", monitor='val_loss')
    ]

    model_history = model_comb.fit(
        x = Xy_train_gen,
        validation_data = Xy_valid,
        steps_per_epoch = steps_per_epoch,
        epochs = N_EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = model_callbacks
        )

    model_comb.save(f"models/comb_{today_label}", overwrite=False)

    with open(f"models/comb_history{today_label}.pk", "wb") as file:
        pickle.dump(model_history, file)

if __name__ == "__main__":
    run()