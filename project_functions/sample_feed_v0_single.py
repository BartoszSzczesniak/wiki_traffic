import numpy as np
from typing import Dict, Tuple

class SampleFeed():
    
    def __init__(self, training_window_size: int, predicted_window_size: int, samples_per_epoch: int, n_features: int) -> None:
        """
        Tool generating inputs and outputs for the single-input version of the wiki traffic model.

        Parameters
        ----------
        training_window_size: int
            length of sequences of numbers of visits used as training samples, 
            eg. training_window_size=90 means that the model predicts based on 90-day history 
        predicted_window_size: int
            length of sequences of target values (predicted),
            eg. predicted_window_size=7 means that the model predicts values for following 7-day
        samples_per_epoch: int
            number of random samples generated per sequence
        n_features: int
            number of features used in the model (sequence of numbers of visits is counted as one feature).
        """

        self.training_window_size = training_window_size
        self.predicted_window_size = predicted_window_size
        self.samples_per_epoch = samples_per_epoch
        self.n_features = n_features

    def random_sample_generator(self, features: Dict[str, np.ndarray], samples_per_page: int = 1, shuffle: bool = False, seed: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features into a generator of samples (records) tailored for LSTM input format, 
        enabling generating complex records without keeping huge matrices in memory.
        
        1. Reshapes and joins together different types of features:
        - visits - array of numbers of visits. Used to derive input sequences for LSTM and target values.
        - page - page features. Time independent variables, repeated for each point in time.
        - time - time features. Time dependent variables.

        2. Generate random sub-sequences of visits array with corresponding target sequences.

        Parameters
        ----------
        features: Dict[str, ndarray]
            dictionary of features to generate lstm samples
        samples_per_page: int
            indicates number of samples derived from each sequence of visits.
        shuffle: bool
            shuffle pages before generating samples
        seed: float | None
            seed for random operations
        """
        page_nos = np.array(
            range(features['visits'].shape[0])
            )
        
        if seed:
            np.random.seed(seed)

        if shuffle:
            np.random.shuffle(page_nos)

        for page_no in page_nos:
        
            # randomly select window

            page_visits = features['visits'][page_no, :]
            notnan = np.where(~np.isnan(page_visits))

            random_window_start = np.random.randint(
                low=np.min(notnan),
                high=np.max(notnan) - self.training_window_size - self.predicted_window_size,
                size=samples_per_page
            )

            # repeat page features for each point in time

            sample_page_features = features['page'][page_no]            
            sample_page_features = np.tile(sample_page_features, self.training_window_size). \
                reshape(self.training_window_size, features['page'].shape[1])

            for window_start in random_window_start:

                window_end = window_start + self.training_window_size     

                # extract number of visits for selected window
                sample_page_visits = page_visits[window_start: window_end]

                # extract time features for selected window
                sample_time_features = features['time'][window_start: window_end, 0:2]
                
                sample_features_all = np.concatenate([
                    sample_page_visits.reshape(-1, 1),
                    sample_time_features,
                    sample_page_features
                    ], axis=1)

                X = sample_features_all.reshape(1, self.training_window_size, self.n_features)
                y = page_visits[window_end: window_end + self.predicted_window_size].reshape(1, self.predicted_window_size)

                yield X, y

    def random_sample_stream(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an infinite stream of generated random samples.  
        See method: random_sample_generator
        """
        while True:
            for X, y in self.random_sample_generator(features, self.samples_per_epoch, True):
                yield X, y

    def random_sample_array(self, features: Dict[str, np.ndarray], samples_per_page: int = 1, shuffle: bool = False, seed: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array of generated random samples.
        See method: random_sample_generator
        """

        Xy = list(
            self.random_sample_generator(
                features, samples_per_page, shuffle, seed)
                )

        X = np.stack([X.reshape(X.shape[1], X.shape[2]) for X, _ in Xy])
        y = np.stack([y.reshape(y.shape[1]) for _, y in Xy])

        return X, y