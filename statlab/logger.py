import logging


def create_logger(name, level=logging.DEBUG, drop: bool = False):
    if name in logging.Logger.manager.loggerDict.keys():
        if drop:
            logging.Logger.manager.loggerDict.pop(name)
        else:
            return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
