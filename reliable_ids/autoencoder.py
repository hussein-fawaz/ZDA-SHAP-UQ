import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class Autoencoder(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            Dense(256, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dense(128, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(64, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(32, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(16, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dense(8, activation=None),
            BatchNormalization(),
            Activation('relu')
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            Dense(8, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dense(16, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(32, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(64, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(128, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dense(256, activation=None),
            BatchNormalization(),
            Activation('relu'),
            Dense(self.input_dim, activation='tanh')
        ])

        # Build the combined encoder-decoder model
        inputs = Input(shape=(self.input_dim,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.full = Model(inputs, decoded)
        self.full.compile(optimizer=Adam(), loss='mse')

    def train(self, X_train, X_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.full.fit(
            X_train, X_train,
            shuffle=True,
            validation_data=(X_val, X_val),
            epochs=10,
            batch_size=512,
            verbose=1,
            callbacks=[early_stopping]
        )
