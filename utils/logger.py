import logging
import json
loggers = {}

APP_NAME = "Supervised Sim Siam"

def get_logger(name=None):
    """
    loggerの設定
    nameが同じすでに設定されているloggerは辞書から返す
    新しいnameのloggerは、新たに設定する。
    """
    global loggers
    if name is None:
        name = APP_NAME

    if loggers.get(name):
        return loggers.get(name)

    logger = logging.getLogger(name)
    # formatter = logging.Formatter('%(levelname)-8s: %(asctime)s | %(filename)-12s - %(funcName)-12s : %(lineno)-4s -- %(message)s',
    #                              datefmt='%Y-%m-%d %H:%M:%S')
    jsonFormat = json.dumps(dict(
        name="%(name)s",
        level="%(levelname)s",
        time="%(asctime)s",
        filename="%(filename)s",
        function="%(funcName)s",
        message="%(message)s"
    ))
    formatter = logging.Formatter(fmt=jsonFormat, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    loggers[name] = logger

    return logger