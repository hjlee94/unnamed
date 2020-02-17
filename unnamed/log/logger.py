from datetime import datetime
import os

_LOG_DIRECTORY = 'logs'
_LOGGING_FORMAT = "%s [%s] %s\n"

class Level:
    DEBUG = 'DEBUG'
    WARN = 'WARN'
    ERROR = 'ERROR'
    INFO = 'INFO'

class Logger:
    s_instance = None
    SCREEN_LOG_ENABLE = True

    @staticmethod
    def get_instance():
        if Logger.s_instance is None:
            Logger.s_instance = Logger()

        return Logger.s_instance

    def __init__(self):
        home_path = os.environ.get('MC_HOME', '.')

        self._log_dir = os.path.join(home_path, _LOG_DIRECTORY)
        self._check_directory()

    def _check_directory(self):
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

    def _log(self, level, message):
        filename = "%s.log"%(datetime.now().strftime('%Y-%m-%d'))
        log_path = os.path.join(self._log_dir, filename)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        log_message = _LOGGING_FORMAT%(timestamp, level, message)

        log_file = open(log_path, 'a+')
        log_file.write(log_message)

        if Logger.SCREEN_LOG_ENABLE:
            print(log_message, end='')

        log_file.close()

    def log_d(self, message):
        self._log(Level.DEBUG, message)

    def log_w(self, message):
        self._log(Level.WARN, message)

    def log_e(self, message):
        self._log(Level.ERROR, message)

    def log_i(self, message):
        self._log(Level.INFO, message)

if __name__ == '__main__':
    logger = Logger.get_instance()
    logger.log_d('Hey')
