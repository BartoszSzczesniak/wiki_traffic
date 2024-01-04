import numpy as np
from typing import Tuple, Dict

from keras import Sequential
from keras import layers
from keras import losses
from keras import callbacks
from keras import regularizers
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
import keras_tuner
import logging

logging.basicConfig(level=logging.INFO)

class LSTMHypermodel(keras_tuner.HyperModel):

    def __init__(
            self, 
            training_window_size, 
            predicted_window_size,
            n_features
            ):
        
        super().__init__()

        self.training_window_size = training_window_size
        self.predicted_window_size = predicted_window_size
        self.n_features = n_features

    def set_hp(self, hp: keras_tuner.HyperParameters):
        """Configure hyperparameters to tune."""

        # hp.Int(name='batch_size', min_value=8, max_value=128, step=2, sampling='log', default=64)
        # hp.Int(name='batch_size', min_value=8, max_value=16, step=2, sampling='log', default=64)
        hp.Fixed(name='batch_size', value=8)
        
        # hp.Int(name='samples_per_epoch', min_value=1, max_value=4, step=2, sampling='log', default=1)
        # hp.Int(name='samples_per_epoch', min_value=2, max_value=4, step=2, sampling='log', default=1)
        hp.Fixed(name='samples_per_epoch', value=4)

        # hp.Int(name='lstm_units', min_value=16, max_value=64, step=2, sampling='log', default=16)
        # hp.Int(name='lstm_units', min_value=32, max_value=64, step=2, sampling='log', default=16)
        hp.Fixed(name='lstm_units', value=64)
        
        hp.Float(name='lstm_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.0)
        # hp.Fixed(name='lstm_dropout', value=0.0)

        hp.Float(name='lstm_l2_recurrent', min_value=0.0, max_value=0.01, step=0.01, default=0.0)

        # hp.Float(name='lstm_l2_kernel', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
        hp.Fixed(name='lstm_l2_kernel', value=0.0)

        hp.Float(name='lstm_l2_bias', min_value=0.0, max_value=0.01, step=0.01, default=0.0)

        # hp.Boolean(name='lstm2_include', default=False)
        hp.Fixed(name='lstm2_include', value=True)

        with hp.conditional_scope(parent_name='lstm2_include', parent_values=[True,]):
        
            # hp.Int(name='lstm2_units', min_value=8, max_value=32, step=2, sampling='log', default=8, parent_name='lstm2_include', parent_values=[True,])
            hp.Fixed(name='lstm2_units', value=32)
            
            hp.Float(name='lstm2_dropout', min_value=0.0, max_value=0.3, step=0.1, default=0.0)
            # hp.Fixed(name='lstm2_dropout', value=0.0)
            
            hp.Float(name='lstm2_l2_recurrent', min_value=0.0, max_value=0.01, step=0.01, default=0.0)

            # hp.Float(name='lstm2_l2_kernel', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
            hp.Fixed(name='lstm2_l2_kernel', value=0.0)
            
            hp.Float(name='lstm2_l2_bias', min_value=0.0, max_value=0.01, step=0.01, default=0.0)

        # hp.Boolean(name='dense1_include', default=False)
        hp.Fixed(name='dense1_include', value=True)

        with hp.conditional_scope(parent_name='dense1_include', parent_values=[True,]):
    
            # hp.Int(name='dense1_units', min_value=16, max_value=64, step=2, sampling='log', default=32)
            # hp.Int(name='dense1_units', min_value=8, max_value=32, step=2, sampling='log', default=32)
            hp.Fixed(name='dense1_units', value=16)
            
            # hp.Boolean(name='dense1_dropout', default=False)
            hp.Fixed(name='dense1_dropout', value=True)
            hp.Float(name='dense1_dropout_rate', min_value=0.0, max_value=0.3, step=0.1, default=0.1, parent_name='dense1_dropout', parent_values=[True,])

            # hp.Float(name='dense1_l2_kernel', min_value=0.0, max_value=0.01, step=0.01, default=0.0)
            hp.Fixed(name='dense1_l2_kernel', value=0.0)

            hp.Float(name='dense1_l2_bias', min_value=0.0, max_value=0.01, step=0.01, default=0.0)

            # hp.Fixed(name='dense1_dropout_rate', value=0.0)
            # hp.Boolean(name='dense2_include', default=False)
            hp.Fixed(name='dense2_include', value=False)

            with hp.conditional_scope(parent_name='dense2_include', parent_values=[True,]):

                hp.Int(name='dense2_units', min_value=8, max_value=16, step=2, sampling='log', default=16)
                hp.Boolean(name='dense2_dropout', default=False)
                hp.Float(name='dense2_dropout_rate', min_value=0.1, max_value=0.3, step=0.1, default=0.1, parent_name='dense2_dropout', parent_values=[True,])
        
        # hp.Float(name='lr', min_value=1e-4, max_value=1e-3, step=10, sampling='log', default=1e-4)
        hp.Fixed(name='lr', value=1e-3)
    
    def build(self, hp: keras_tuner.HyperParameters):

        self.set_hp(hp)
                
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(self.training_window_size, self.n_features)))
        model.add(
            layers.LSTM(
                units = hp.get('lstm_units'), 
                return_sequences = hp.get('lstm2_include'),
                recurrent_regularizer = regularizers.L2(hp.get('lstm_l2_recurrent')),
                kernel_regularizer = regularizers.L2(hp.get('lstm_l2_kernel')),
                bias_regularizer = regularizers.L2(hp.get('lstm_l2_bias')),
                recurrent_dropout = hp.get('lstm_dropout'),
                ))
        
        if hp.get('lstm2_include'):
            model.add(layers.LSTM(
                units = hp.get('lstm2_units'), 
                return_sequences=False, 
                recurrent_regularizer=regularizers.L2(hp.get('lstm2_l2_recurrent')),
                kernel_regularizer=regularizers.L2(hp.get('lstm2_l2_kernel')),
                bias_regularizer=regularizers.L2(hp.get('lstm2_l2_bias')),
                recurrent_dropout = hp.get('lstm2_dropout'),
                ))

        if hp.get('dense1_include'):
            model.add(layers.Dense(
                units = hp.get('dense1_units'), 
                activation = 'relu',
                kernel_regularizer=regularizers.L2(hp.get('dense1_l2_kernel')),
                bias_regularizer=regularizers.L2(hp.get('dense1_l2_bias'))
                ))
        
        if hp.is_active('dense1_dropout') and hp.get('dense1_dropout'):
            model.add(layers.Dropout(hp.get('dense1_dropout_rate')))

        if hp.is_active('dense2_include') and hp.get('dense2_include'):
            model.add(layers.Dense(hp.get('dense2_units'), 'relu'))
            
        if hp.is_active('dense2_dropout') and hp.get('dense2_dropout'):
            model.add(layers.Dropout(hp.get('dense2_dropout_rate')))
        
        model.add(layers.Dense(self.predicted_window_size, 'sigmoid'))

        model.compile(
            loss = losses.Huber(0.25), 
            optimizer = Adam(learning_rate = hp.get('lr')), 
            metrics = RootMeanSquaredError())

        return model
    
    def fit(self, hp: keras_tuner.HyperParameters, model, features_train, features_test, epochs, *args, **kwargs):

        Xy_train = self.preprocess_train_dataset(hp, features_train)
        Xy_test = self.preprocess_test_dataset(hp, features_test)
        
        assert Xy_test[0].shape[0] == Xy_test[1].shape[0]
        assert Xy_test[0].shape[1] == self.training_window_size
        assert Xy_test[0].shape[2] == self.n_features
        assert Xy_test[1].shape[1] == self.predicted_window_size

        n_rows = features_train['visits'].shape[0]

        self.steps_per_epoch = round(n_rows * hp.get('samples_per_epoch') / hp.get('batch_size'))

        return model.fit(
            x = Xy_train,
            validation_data = Xy_test,
            batch_size = hp.get('batch_size'),
            steps_per_epoch = self.steps_per_epoch,
            epochs = epochs,
            *args,
            **kwargs
        )
    
    def preprocess_train_dataset(self, hp: keras_tuner.HyperParameters, features_train: Dict[str, np.ndarray]):
        """Prepare stream of preprocessed samples for training (with tunable parameters)"""

        return self.random_sample_stream(
            hp,
            features_train,
            samples_per_iter = hp.get('samples_per_epoch')
            )
    
    def preprocess_test_dataset(self, hp: keras_tuner.HyperParameters, features_test: Dict[str, np.ndarray]):
        """Prepare preprocessed test dataset"""

        Xy_test_gen = self.random_sample_generator(
            hp,
            features_test,
            samples_per_page = 1,
            shuffle = False,
            seed = 0
            )

        Xy_test = list(Xy_test_gen)
        X_test = np.stack([X.reshape(self.training_window_size, self.n_features) for X, _ in Xy_test])
        y_test = np.stack([y for _, y in Xy_test])

        return X_test, y_test
    
    def random_sample_generator(self, hp: keras_tuner.HyperParameters, features: Dict[str, np.ndarray], samples_per_page: np.int64 = 1, shuffle: bool = False, seed = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generator of training samples (see: sample_feed.py)"""

        page_nos = np.array(
            range(features['visits'].shape[0])
            )
        
        if seed:
            np.random.seed(seed)

        if shuffle:
            np.random.shuffle(page_nos)

        for page_no in page_nos:
        
            page_visits = features['visits'][page_no, :]
            notnan = np.where(~np.isnan(page_visits))

            random_window_start = np.random.randint(
                low=np.min(notnan),
                high=np.max(notnan) - self.training_window_size - self.predicted_window_size,
                size=samples_per_page
            )

            sample_page_features = features['page'][page_no]
            
            sample_page_features = np.tile(sample_page_features, self.training_window_size). \
                reshape(self.training_window_size, features['page'].shape[1])

            for window_start in random_window_start:
                window_end = window_start + self.training_window_size
                
                sample_page_visits = page_visits[window_start: window_end]
                
                sample_time_features = features['time'][window_start: window_end]
                
                sample_features_all = np.concatenate([
                    sample_page_visits.reshape(-1, 1),
                    sample_time_features,
                    sample_page_features
                    ], axis=1)

                sample_features_all = sample_features_all.reshape(1, self.training_window_size, self.n_features)
                
                sample_target = page_visits[window_end: window_end + self.predicted_window_size]

                yield sample_features_all, sample_target

    def random_sample_stream(self, hp: keras_tuner.HyperParameters, features: Dict[str, np.ndarray], samples_per_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stream of training samples (see: sample_feed.py)"""

        while True:
            for X, y in self.random_sample_generator(hp, features, samples_per_iter, True):
                yield X, y

TRAINING_WINDOW_SIZE = 90
PREDICTED_WINDOW_SIZE = 7

N_FEATURES = 21
N_EPOCHS = 6

MAX_TRIALS = 10
REDUCE_PAGES = 10
PROJECT_NAME = "try1220"

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
        n_features = N_FEATURES
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
        features_test = features_valid,
        epochs = N_EPOCHS,
        callbacks = model_callbacks
        )
    
if __name__ == "__main__":
    run()