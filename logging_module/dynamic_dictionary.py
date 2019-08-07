from __future__ import absolute_import, print_function, unicode_literals
import json
import os

from . import constants


def config_initial(config_file_path):
    """
    Loads the config.json file with basic initial configuration.
    Creates a 'root.log' file for root logger.
    :param config_file_path: path
    :return: configuration_dictionary
    """
    log_setting = {
        'version': 1,

        'handlers': {
            'hand_root': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'form_root',
                'filename': os.path.join(os.getcwd(), "logs", 'root.log'),
                'mode': 'a',
                'maxBytes': 10240000,
                'backupCount': 5
            },
        },

        'formatters': {
            'form_root': {
                'format': '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
                'class': 'logging.Formatter'
            }
        },

        'loggers': {

        },

        'root': {
            'level': 'INFO',
            'handlers': ['hand_root', ]
        }
    }

    # Saving Dictionary to json file
    try:
        with open(config_file_path, 'w') as config:
            json.dump(log_setting, config)
    except ValueError as e:
          print ("Error Occurred : {0}".format(str(e)))
          exit()
    else:
        # Return the dictionary
        return log_setting


def update_config_dictionary(logger_name, base_path, config_file_path, handler_type, logging_level, host,
                             port, rotating_maxyBytes, rotating_backupCount, timed_rotating_when,
                             timed_rotating_interval, timed_rotating_backupCount):
    """
    Updates the config.json file and adds the configuration for the new logger instance.
    Reads the config.json first to fetch the dictionary and then appends the new information in it
    and writes it back to config.json file.

    :param logger_name: name of the logger
    :param base_path: path where log record will be created
    :param config_file_path: path where the config.json is stored
    :param handler_type: type of the handler to be associated with logger instance
    :param logging_level: Level of logging required [ OPTIONAL --> DEFAULT: INFO]
    :param host: optional [to be used only with network handlers like SocketHandler]
    :param port: optional [to be used only with network handlers like SocketHandler]
    :return: Nothing
    """

    options_dictionary = dict()  # For Handlers
    logger_options_dictionary = dict()  # For loggers

    # Retrieve the dictionary
    fp = open(config_file_path, 'r')
    config_dictionary = dict(json.load(fp))
    fp.close()

    handler_name = 'hand' + '_' + logger_name + '_' + handler_type

    if handler_type == constants.CONST_STREAM_HANDLER:
        """
        code for StreamHandler
        """

    elif handler_type == constants.CONST_FILE_HANDLER:
        """
        """

    elif handler_type == constants.CONST_NULL_HANDLER:
        """
            """

    elif handler_type == constants.CONST_WATCHED_FILE_HANDLER:
        """
            """

    elif handler_type == constants.CONST_ROTATING_FILE_HANDLER:
        options_dictionary["class"] = 'logging.handlers.RotatingFileHandler'
        options_dictionary["level"] = logging_level
        options_dictionary["formatter"] = 'form_root'
        options_dictionary["filename"] = os.path.join(base_path, logger_name + '.log')  # Different according to logger
        options_dictionary["mode"] = 'a'
        options_dictionary["maxBytes"] = rotating_maxyBytes
        options_dictionary["backupCount"] = rotating_backupCount

    elif handler_type == constants.CONST_TIMED_ROTATING_FILE_HANDLER:
        options_dictionary["class"] = 'logging.handlers.TimedRotatingFileHandler'
        options_dictionary["level"] = logging_level
        options_dictionary["formatter"] = 'form_root'
        options_dictionary["filename"] = os.path.join(base_path, logger_name + '.log')  # Different according to logger
        options_dictionary["when"] = timed_rotating_when
        options_dictionary["interval"] = timed_rotating_interval
        options_dictionary["backupCount"] = timed_rotating_backupCount

    elif handler_type == constants.CONST_SOCKET_HANDLER:
        options_dictionary["class"] = 'logging.handlers.SocketHandler'
        options_dictionary["level"] = logging_level
        options_dictionary["formatter"] = 'form_root'
        options_dictionary["host"] = host
        options_dictionary["port"] = port

    elif handler_type == constants.CONST_DATAGRAM_HANDLER:
        """
            """

    elif handler_type == constants.CONST_SYSLOG_HANDLER:
        """
            """

    elif handler_type == constants.CONST_NT_EVENT_LOG_HANDLER:
        """
            """

    elif handler_type == constants.CONST_SMTP_HANDLER:
        """
            """

    elif handler_type == constants.CONST_MEMORY_HANDLER:
        """
            """

    elif handler_type == constants.CONST_HTTP_HANDLER:
        """
            """

    # Add the new handler to config_dictionary
    (config_dictionary['handlers'])[handler_name] = options_dictionary
    # print (config_dictionary['handlers'])[handler_name]

    # keep the formatter same -- as same format for each and every logger

    # Retrieving list of available loggers in config dictionary
    loggers_list = list((config_dictionary['loggers']).keys())


    if logger_name in loggers_list:
        """
        logger available -- only append the new handler name
        """
        handlers_list = ((config_dictionary['loggers'])[logger_name])['handlers']  # Fetching the current handlers list
        handlers_list.append(handler_name)  # Appending the new handler name

        print ('Handlers associated with ' + logger_name + ': ', handlers_list)
        ((config_dictionary['loggers'])[logger_name])['handlers'] = handlers_list  # Updating the logger's configuration

    else:
        """
        No logger available - create a new one and update the configuration
        """
        logger_options_dictionary['level'] = logging_level
        logger_options_dictionary['handlers'] = [handler_name, ]
        logger_options_dictionary['propagate'] = 0
        logger_options_dictionary['qualname'] = 'xyz'
        (config_dictionary['loggers'])[logger_name] = logger_options_dictionary

    # print config_dictionary

    # Saving Dictionary to json file
    try:
        with open(config_file_path, 'w') as config:
            json.dump(config_dictionary, config)
    except ValueError as e:
         print ("Error Occurred : {0}".format(str(e)))
         exit()
    else:
        return 0