"""
File name: logging_module.py
Author: Jitendra Saxena
Date created: 4/07/2017
Date last modified:
Python Version: 2.7
Description: Additional module for creating logger instances and saving their configuration
             in the configuration_dictionary. Specify only the handler type and the
             level_of_logging [OPTIONAL DEFAULT:INFO]
             and the rest of configuration will be done by this module.

             Returns the desired logger instance.

"""
from __future__ import absolute_import, print_function, unicode_literals
import json
import logging
import logging.config
import logging.handlers
import os
from settings import MODEL_DIRECTORY

from . import constants
from . import dynamic_dictionary

# file = open('app.py')
# task = file.readline().split(' ')[1]

current_directory = os.path.join(os.getcwd(), MODEL_DIRECTORY.split('_')[0] + '_app_logs')
print ("++++ Current working directory for logging module:", current_directory)
# base_path = os.path.join(current_directory, "logs", str(datetime.datetime.now().strftime('%Y')),
#                          str(datetime.datetime.now().strftime('%B')), str(datetime.datetime.now().strftime('%d')))
base_path = os.path.join(current_directory, "logs")
# Configuration_File path
config_file_path = os.path.join(current_directory, "config", "config.json")

# Mapping List file path
mapping_list_path = os.path.join(
    current_directory, "mapping-list", "mapping_list.json")
# .gitignore file path
git_ignore_file = os.path.join(os.getcwd(), ".gitignore")
git_file_content = "/logging_module/config/* \n/logging_module/mappinglist/* \n/logging_module/logs/* \n/logging_module/root.log"

# Global Dictionary 'logger_list'
logger_mapping_dictionary = {}
mapping_dictionary_updated = True
previous_config_read = False


def create_and_load_config_file():
    """
    Create a Config folder in root directory of project and a config.json file in it. This file
    will be used as the configuration file.

    Config.json will be loaded with initial basic configuration by calling the config_initial
    method of dynamic_dictionary module.

    Returns:
        configuration_dictionary [ which can further be passed to dictConfig() ]
    """
    # Create .gitignore file
    if not os.path.isfile(git_ignore_file):
        open(git_ignore_file, 'a').close()
        with open(git_ignore_file, 'a') as gitfile:
            gitfile.write(git_file_content)

    # Creating Config folder
    config_folder = os.path.join(current_directory, "config")
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Config.json file creation when not there
    if not os.path.isfile(config_file_path):
        open(config_file_path, 'a').close()

        # Always delete the existing mapping_list when creating the new
        # config.json file
        if os.path.isfile(mapping_list_path):
            print ('**** Deleting existing mapping_list ****')
            os.remove(mapping_list_path)

        # Load the initial configuration in the 'Config_settings' dictionary
        return dynamic_dictionary.config_initial(config_file_path)

    # Returning the config_dictionary if config.json is there
    else:
        try:
            with open(config_file_path, 'r') as fp:
                global previous_config_read
                previous_config_read = True
                return dict(json.load(fp))

        except ValueError as e:
            print ("Error Occurred : {0}".format(str(e)))
            print (">>>>>>config.json --> Configuration file is empty or missing !!")

            # Simply delete the config.json
            # This file will be created automatically on next run
            os.remove(config_file_path)

            # print " ****** Restoring the default Configuration in config.json
            # ******"

            # # Restoring the default
            # check_config = dynamic_dictionary.config_initial(config_file_path)
            # os.remove(mapping_list_path)

            # if len(check_config) != 0:
            #     print "****** Successfully restored ******"
            # else:
            #     print "****** Failed to restore ******"

            print (">>>> Exiting - Restart.....")
            exit()


def create_logger(logger_name, handler_type, logging_level="DEBUG", host=constants.CONST_DEFAULT_HOST_IP,
                  port=constants.CONST_DEFAULT_PORT, rotating_maxyBytes=1024000, rotating_backupCount=100,
                  timed_rotating_when='D', timed_rotating_interval=1, timed_rotating_backupCount=2):
    """
    First searches in a mapping list. If mapping is there for the given logger_name, it means
    we have the configuration already available in configuration_dictionary.

    Otherwise, we will create a new logger instance and set up its configuration.
    Add the configuration in the config.json file.

    :param logger_name: name with which logger is to be created
    :param handler_type: types of handler to be used for the logger [pass as tuple or list]
    :param logging_level: Optional parameter [to pass the level of logging]
                           DEFAULT LEVEL : DEBUG
    :param host: IP address of the destination [ OPTIONAL ] 
                -- change default value from constants.py (constants.CONST_DEFAULT_HOST_IP)
    :param port: port number [ OPTIONAL ] 
                 -- change default value from constants.py (constants.CONST_DEFAULT_PORT)
    :param rotating_maxyBytes: For ROTATING_FILE_HANDLER's maxBytes arg [optional, Default: 102400 Bytes]
    :param rotating_backupCount: For ROTATING_FILE_HANDLER's backupCount arg [optional, Default: 2]
    :param timed_rotating_when: For TIMED_ROTATING_FILE_HANDLER's when arg [optional, Default: 'D']
    :param timed_rotating_interval: For TIMED_ROTATING_FILE_HANDLER's interval arg [optional, Default: 1]
    :param timed_rotating_backupCount: For TIMED_ROTATING_FILE_HANDLER's backupCount arg [optional, Default: 2]

    :return: logger instance
    """

    # logger_list = get_mapping_list()  # Retrieving the mapping list
    # print logger_list
    # if logger_name in logger_list:
    for handler_item in handler_type:
        result = check_logger_and_handler_exist(logger_name, handler_item)

        # If result is true
        if result:
            pass

        else:
            print (">>>>>> Logger Configuration not available ")
            print ("...... Creating Logger ......")
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging_level)
            handler = get_handler(logger_name, handler_item, host, port, rotating_maxyBytes, rotating_backupCount,
                                  timed_rotating_when, timed_rotating_interval, timed_rotating_backupCount)
            formatter = logging.Formatter(
                '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Add the newly created logger instance in config.json file
            dynamic_dictionary.update_config_dictionary(logger_name, base_path, config_file_path, handler_item,
                                                        logging_level, host, port, rotating_maxyBytes,
                                                        rotating_backupCount, timed_rotating_when,
                                                        timed_rotating_interval, timed_rotating_backupCount)

            # Update the dictionary 'logger_list'

            # Fetch the list of handlers associated with a particular logger_name
            # and update the handlers list

            key = logger_name
            value = list()

            # If logger exists in the dictionary, retrieve the already available handlers
            #  then append the new handler_type for the logger
            if key in list(logger_mapping_dictionary.keys()):
                value = logger_mapping_dictionary[key]

            value.append(handler_item)
            # New handler type added to the specific logger's handlers list
            logger_mapping_dictionary[key] = value
            """
            print '#########################'
            print 'Mapping Dictionary now :'
            print logger_mapping_dictionary
            print '#########################'
            """
            # logger_list.append(logger_name)
            # print logger_list

            update_mapping_list()

    logger = logging.getLogger(logger_name)
    return logger


def get_mapping_list():
    """
    Create a file for storing the name of loggers that have been configured.

    If file exists, load the list from file and simply return the list.
    Otherwise create a file and return empty list.

    :return: List containing loggers name.
    """
    mapping_list_folder = os.path.join(current_directory, "mapping-list")

    if not os.path.exists(mapping_list_folder):
        os.makedirs(os.path.join(mapping_list_folder))

    # List for storing the instances
    if not os.path.isfile(mapping_list_path):
        print ('**** Creating mapping dictionary ****')

        # Checking if the previous configuration available, then overwrite it with the default configuration
        # This happens when mapping_list.json doesn't exist and config.json
        # exists with some previous configuration

        if previous_config_read:
            # Overwrite the config.json with initial config
            print ("Overwriting the previous configuration")
            dynamic_dictionary.config_initial(config_file_path)

        # creates a json file for dictionary
        open(mapping_list_path, 'a').close()
        return {}  # Returning the empty dictionary

    else:
        # Load the mapping_list with previous loggers
        print ('**** Loading the mapping dictionary ****')
        try:
            with open(mapping_list_path, 'r') as json_file:
                return json.load(json_file)
        except ValueError as e:
            print ("****** Error Occurred : {0} ******".format(str(e)))
            print (">>>>>> mapping_list.json --> File is empty or missing !!")
            os.remove(mapping_list_path)
            os.remove(config_file_path)
            print (">>>> Exiting - Restart.....")
            exit()


def update_mapping_list():
    """
    Updates the file storing the list with loggers name and the associated handlers
    :return: None
    """
    print ('****** Updating mapping.json ******')
    try:
        with open(mapping_list_path, 'w') as json_file:
            json.dump(logger_mapping_dictionary, json_file)
    except ValueError as e:
        print ("Error Occurred : {0}".format(str(e)))
        exit()
    else:
        # Set mapping_dictionary_updated to TRUE , so that new mapping list can
        # be loaded
        global mapping_dictionary_updated
        mapping_dictionary_updated = True

    return 0


def get_handler(logger_name, handler_type, host, port, rotating_maxyBytes, rotating_backupCount, timed_rotating_when,
                timed_rotating_interval, timed_rotating_backupCount):
    """
    Create handler of appropriate type

    :param logger_name: name of the logger to which handler will be attached
    :param handler_type: type of the handler required
    :param host: optional [to be used only if Network Handlers required]
    :param port: optional [to be used only if Network Handlers required]
    :return: required handler type.
    """
    log_path = os.path.join(base_path, logger_name + ".log")

    if handler_type == "StreamHandler":
        """
        code for StreamHandler
        """

    elif handler_type == constants.CONST_FILE_HANDLER:
        file_handler = logging.FileHandler(log_path, mode='a')
        return file_handler

    elif handler_type == constants.CONST_NULL_HANDLER:
        """
            """

    elif handler_type == constants.CONST_WATCHED_FILE_HANDLER:
        """
            """

    elif handler_type == constants.CONST_ROTATING_FILE_HANDLER:
        rotating_file_handler = logging.handlers.RotatingFileHandler(log_path, 'a', maxBytes=rotating_maxyBytes,
                                                                     backupCount=rotating_backupCount)
        return rotating_file_handler

    elif handler_type == constants.CONST_TIMED_ROTATING_FILE_HANDLER:
        timed_rotating_file_handler = logging.handlers.TimedRotatingFileHandler(log_path, when=timed_rotating_when,
                                                                                interval=timed_rotating_interval,
                                                                                backupCount=timed_rotating_backupCount)
        return timed_rotating_file_handler

    elif handler_type == constants.CONST_SOCKET_HANDLER:
        socket_handler = logging.handlers.SocketHandler(host, port)
        return socket_handler

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


def check_logger_and_handler_exist(logger_name, handler_type):
    """
    Check for the logger and the associated handlers with it.

    Returns TRUE if the logger and the required handler type exists in the dictionary,
    Else returns FALSE if either logger doesn't exist or [if logger exists but the required handler_type doesn't exists]

    :param logger_name: Name of the logger
    :param handler_type: Handler type
    :return: boolean value
    """
    global logger_mapping_dictionary
    global mapping_dictionary_updated

    if mapping_dictionary_updated:
        # Retrieving the mapping list for first time
        logger_mapping_dictionary = get_mapping_list()
        # Mapping dictionary will not be loaded again unless updated
        mapping_dictionary_updated = False

    if logger_name in list(logger_mapping_dictionary.keys()):
        if handler_type in logger_mapping_dictionary[logger_name]:
            return True
        else:
            return False
    else:
        return False
