from typing import Dict, Tuple
import numpy as np

class SampleFeed():
    
    def __init__(self, training_window_size: int, predicted_window_size: int, samples_per_epoch: int) -> None:
        """
        Tool generating inputs and outputs for the multi-input version of the wiki traffic model.

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
        """

        self.training_window_size = training_window_size
        self.predicted_window_size = predicted_window_size
        self.samples_per_epoch = samples_per_epoch

    def random_sample_generator(self, features: Dict[str, np.ndarray], samples_per_page: int = 1, shuffle: bool = False, seed: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features into a generator of samples (records) tailored for the format expected by the model, 
        enabling generating complex records without keeping huge matrices in memory.
        
        1. Reshapes and joins together different types of features:
        - visits - time series of numbers of visits. Used to derive input sequences for LSTM and target values.
        - page - page features. Time independent variables, static for each page.
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

            # extract static page features
            static_page_features = features['page'][page_no]

            for window_start in random_window_start:

                window_end = window_start + self.training_window_size     

                # extract number of visits for selected window
                seq_page_visits = page_visits[window_start: window_end].reshape(-1,1)

                # extract time features for selected window
                seq_time_features = features['time'][window_start: window_end, 0:2]

                # extract static time features (for predicted window)
                static_time_features = features['time'][window_end: window_end + self.predicted_window_size, 0:2].reshape(-1)

                seq_features = np.concatenate([seq_page_visits, seq_time_features], axis=1)
                static_features = np.concatenate([static_time_features, static_page_features])

                X_seq = seq_features.reshape(1, self.training_window_size, -1)
                X_stc = static_features.reshape(1, -1)
                y = page_visits[window_end: window_end + self.predicted_window_size].reshape(1, self.predicted_window_size)

                yield (X_seq, X_stc), y

    def random_sample_stream(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an infinite stream of generated random samples.  
        See method: random_sample_generator
        """
        while True:
            for (X_seq, X_stc), y in self.random_sample_generator(features, self.samples_per_epoch, True):
                yield (X_seq, X_stc), y

    def random_sample_array(self, features: Dict[str, np.ndarray], samples_per_page: int = 1, shuffle: bool = False, seed: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an array of generated random samples.
        See method: random_sample_generator
        """

        Xy = list(
            self.random_sample_generator(
                features, samples_per_page, shuffle, seed)
                )
        
        X_seq = np.stack([X_seq.reshape(X_seq.shape[1], X_seq.shape[2]) for (X_seq, _), _ in Xy])
        X_stc = np.stack([X_stc.reshape(X_stc.shape[1]) for (_, X_stc), _ in Xy])
        y = np.stack([y.reshape(y.shape[1]) for (_, _), y in Xy])

        return (X_seq, X_stc), y