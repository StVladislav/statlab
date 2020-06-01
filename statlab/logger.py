import logging


def create_logger(
        name,
        level_console=logging.DEBUG,
        drop: bool = False,
        path_to_save: str = None,
        level_for_file=logging.WARNING
):
    if name in logging.Logger.manager.loggerDict.keys():
        if drop:
            logging.Logger.manager.loggerDict.pop(name)
        else:
            return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level_console)

    ch = logging.StreamHandler()
    ch.setLevel(level_console)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if path_to_save is not None:
        fh = logging.FileHandler(path_to_save)
        fh.setLevel(level_for_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
