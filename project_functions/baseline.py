import tensorflow as tf

class NaiveModel(tf.keras.Model):
    def __init__(self, predicted_window_size):
        super().__init__()
        self.predicted_window_size = predicted_window_size

    def call(self, inputs):
        return tf.repeat(inputs[:, -1:, 0], self.predicted_window_size, axis=1)


# # notes - attempts towards ARIMA
# Might be used as a benchmark

# from darts.models import AutoARIMA
# from darts import TimeSeries

# i = 1

# df_i = df_wiki_T.iloc[:, i]
# df_i = df_i[df_i.index <= df_i[df_i != 0.0].index.max()]

# ts_all = TimeSeries.from_series(df_i)

# ts_train, ts_test = ts_all.split_before(0.9)

# model = AutoARIMA()
# model.fit(ts_train)

# ts_pred = model.predict(len(ts_test))

# df_i.plot(label="actual")
# ts_pred.plot(label="forecast", lw=3)
# plt.title(df_wiki_T.columns[i])

