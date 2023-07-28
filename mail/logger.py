import logging as log


def setupLogging(level=log.INFO):
    log.basicConfig(level=log.INFO, format="(%(asctime)s) [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
