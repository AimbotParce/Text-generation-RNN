import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from vocab import getVocab
import pathlib
from art import text2art
from training import MailModel


CHECKPOINTFOLDER = pathlib.Path(__file__).parent.parent / "checkpoints" / "mail"

vocab = getVocab()
word2id = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
id2word = tf.keras.layers.StringLookup(vocabulary=word2id.get_vocabulary(), invert=True, mask_token=None)


class SingleStepModel(tf.keras.Model):
    def __init__(self, model, word2id, id2word, temperature=1.0):
        super().__init__()
        self.model = model
        self.word2id = word2id
        self.id2word = id2word
        self.temperature = temperature

    @tf.function
    def generate(self, prompt, states=None):
        prompt = tf.strings.regex_replace(prompt, "\n", " ")
        prompt = tf.strings.regex_replace(prompt, "\t", " ")
        prompt = tf.strings.regex_replace(prompt, "([^\w\s])", r" \1 ")
        prompt = tf.strings.regex_replace(prompt, " +", " ")
        prompt = tf.strings.lower(prompt)
        prompt = tf.strings.split(prompt, " ")

        prompt = word2id(prompt)

        prompt = tf.expand_dims(prompt, 0)

        predictions, states = self.model(inputs=prompt, states=states, return_state=True)

        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / self.temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=-1)
        return id2word(predicted_id), states


def format(prompt, response):
    phrase = " ".join([prompt] + response)
    # Remove spaces before punctuation
    phrase = tf.strings.regex_replace(phrase, " ([^\w\s])", r"\1")
    return phrase


def printResponse(response):
    phrase = format("", response).numpy().decode("utf-8")

    # Clear line
    print("\033[K", end="")
    print(phrase, end="", flush=True)


def main():
    best = None
    for model in CHECKPOINTFOLDER.glob("*.index"):
        if best is None or model.stat().st_mtime > best.stat().st_mtime:
            best = model

    model = MailModel(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024)
    model.load_weights(best.with_suffix(""))

    model = SingleStepModel(model, word2id, id2word)

    print(text2art("Welcome to Mail GPT!"))
    states = None
    while True:
        prompt = input("[IN] > ")

        response = []
        for _ in range(100):
            phrase = format(prompt, response)
            word, states = model.generate(phrase, states=states)
            # Word is a tensor, so we need to convert it to a string
            word = word.numpy()[0].decode("utf-8")
            response.append(word)
            printResponse(response)
        print()


if __name__ == "__main__":
    main()
