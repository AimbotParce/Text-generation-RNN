import tensorflow as tf
import os
from tqdm import tqdm
import logging as log
import json
import email

MAILSIZE = 100  # Number of words in a mail
DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data", "mails")
MAILPATH = os.path.join(DATAPATH, "enron_mails")


def getDataset() -> tuple[tf.data.Dataset, int]:
    log.info("Computing mail count...")
    mailCount = 0
    for root, dirs, files in os.walk(MAILPATH):
        for file in files:
            mailCount += 1

    log.info("Mail count: %d", mailCount)

    # Mail Reading generator:
    def mail_generator(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), "r", errors="ignore") as f:
                    mail = f.read()
                    mail = email.message_from_string(mail)
                    body = mail.get_payload()
                    if body is None:
                        continue
                    yield body

    mailDataset = tf.data.Dataset.from_generator(
        mail_generator, args=[MAILPATH], output_types=tf.string, output_shapes=()
    )

    # Remove \n from mails
    mailDataset = mailDataset.map(lambda mail: tf.strings.regex_replace(mail, "\n", " "))
    # Remove tabs
    mailDataset = mailDataset.map(lambda mail: tf.strings.regex_replace(mail, "\t", " "))

    # Split and Remove everything starting from -----Original Message----- and things like that
    mailDataset = mailDataset.map(lambda mail: tf.strings.split(mail, "-----", maxsplit=1)[0])

    # Substitute all punctuation with the same punctuation + a space (so that it becomes a word)
    mailDataset = mailDataset.map(lambda mail: tf.strings.regex_replace(mail, "([^\w\s])", r" \1 "))

    # Remove nth spaces
    mailDataset = mailDataset.map(lambda mail: tf.strings.regex_replace(mail, " +", " "))

    # Make everything lowercase
    mailDataset = mailDataset.map(lambda mail: tf.strings.lower(mail))

    # Skip empty mails
    mailDataset = mailDataset.filter(lambda mail: tf.strings.length(mail) > 0)

    # Split mails into words
    mailDataset = mailDataset.map(lambda mail: tf.strings.split(mail, " "))

    # Drop mails that are too short
    mailDataset = mailDataset.filter(lambda mail: tf.shape(mail)[0] > MAILSIZE)

    return mailDataset, mailCount
