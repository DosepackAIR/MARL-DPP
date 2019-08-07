import logging

from logging_module import constants
from logging_module import logging_module


def setup_logger(logger_name, log_file=None, level='DEBUG'):
    config = logging_module.create_and_load_config_file()
    logging.config.dictConfig(config)
    my_logger = logging_module.create_logger(logger_name,
                                             [constants.CONST_SOCKET_HANDLER, constants.CONST_ROTATING_FILE_HANDLER],
                                             logging_level=level,
                                             host='localhost', port=9020)
    return my_logger
