import tensorflow as tf
import os
from tqdm import tqdm
import logging as log
import json
from dataset import getDataset, DATAPATH
from logger import setupLogging


def createVocab():
    mailDataset, mailCount = getDataset()
    # Create a vocab of all words. We'll suppose that punctuations are words themselves
    vocab = {}
    for mail in tqdm(mailDataset, total=mailCount, desc="Computing vocab"):
        for word in mail.numpy():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Create a list of words sorted by frequency
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    # Remove words that appear less than 5 times
    vocab = list(filter(lambda x: x[1] > 5, vocab))
    print("Vocab size:", len(vocab))

    with open(os.path.join(DATAPATH, "vocab.txt"), "w") as f:
        for word, frequency in vocab:
            word = word.decode("utf-8")
            f.write(word + "\t" + str(frequency) + "\n")


def getVocab():
    with open(os.path.join(DATAPATH, "vocab.txt"), "r") as f:
        vocab = []
        for line in f.readlines():
            word, frequency = line.split("\t")
            vocab.append(word)
        return vocab


if __name__ == "__main__":
    setupLogging()
    createVocab()
