import logging_module
import logging
import logging.config
import logging.handlers
import constants

logging_level = ['INFO', 'DEBUG', 'WARNING', 'CRITICAL', 'ERROR']


def main():
    config = logging_module.create_and_load_config_file()
    # print config
    logging.config.dictConfig(config)

    my_logger = logging_module.create_logger('robot5', [constants.CONST_TIMED_ROTATING_FILE_HANDLER, constants.CONST_SOCKET_HANDLER], logging_level='DEBUG',
                                             host='localhost', port=9020)

    # my_logger1 = logging_module.create_logger('robot_10', [constants.CONST_SOCKET_HANDLER, constants.CONST_TIMED_ROTATING_FILE_HANDLER], logging_level='DEBUG',
                                           # host=constants.CONST_DEFAULT_HOST_IP, port=constants.CONST_DEFAULT_PORT)
    # my_logger = logging_module.create_logger('JKS1', CONST_ROTATING_FILE_HANDLER)
    print '******* Logger created successfully *******'
    # return
    for i in range(1, 6, 1):
        my_logger.info("Hey there, I am going great" + str(i))
       # my_logger1.info("Hey there, I am going great" + str(i))
    print '******* Records thrown successfully *******'


if __name__ == "__main__":
    main()
