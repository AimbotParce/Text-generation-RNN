import tensorflow as tf
import os
from tqdm import tqdm
import logging as log
import json
from logger import setupLogging
from dataset import getDataset, MAILSIZE
from vocab import getVocab

CHECKPOINTFOLDER = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mail")


class MailModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def main():
    setupLogging()

    mailDataset, mailCount = getDataset()
    vocab = getVocab()

    word2id = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)

    # Convert mails to ids
    mailDataset = mailDataset.map(lambda mail: word2id(mail))

    # Cut mails to a fixed size
    mailDataset = mailDataset.map(lambda mail: mail[:MAILSIZE])

    # Split mail into input and target
    mailDataset = mailDataset.map(split_input_target)

    model = MailModel(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINTFOLDER, "model_{epoch}"), save_weights_only=True
    )

    model.fit(
        mailDataset.shuffle(1000).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE),
        epochs=30,
        callbacks=[modelCheckpoint],
    )


if __name__ == "__main__":
    main()


# TO TEST THE MODEL
# for input_example_batch, target_example_batch in mailDataset.batch(1).take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
#     # Extract a mail:
#     sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#     sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

#     email = ""
#     for index in sampled_indices:
#         email += id2word(index) + " "

#     print(email)
