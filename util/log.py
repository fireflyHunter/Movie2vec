import sys

import logbook
from logbook import StreamHandler, Logger

from .singles import Singleton

_author__ = "Aonghus Lawlor"
__copyright__ = "Copyright (c) 2015"
__credits__ = ["Aonghus Lawlor", "Khalil Muhammad", "Ruhai Dong"]
__license__ = "All Rights Reserved"
__version__ = "1.0.0"
__maintainer__ = "Aonghus Lawlor"
__email__ = "aonghus.lawlor@insight-centre.org"
__status__ = "Development"

class LoggerSingle(Logger):
    __metaclass__ = Singleton


#file_handler = logbook.RotatingFileHandler(config.LOGFILE, level=logbook.DEBUG)
#console_handler = logbook.StreamHandler(sys.stdout, level=logbook.INFO, bubble=True)
#file_handler.push_application()
#console_handler.push_application()

# log_format = (
#     u'[{record.time:%Y-%m-%d %H:%M:%S.%f} pid({record.process})] ' +
#     u'{record.level_name}: {record.module}::{record.func_name}:{record.lineno} {record.message}'
# )
# #default_handler = StderrHandler(format_string=log_format)
# default_handler = StreamHandler(sys.stdout, format_string=log_format)
# default_handler.push_application()

def get_logger(log_format=None):
    """
    Return the logger for the given name.
    :param name: The name of the logger.
    :return: A logbook Logger.
    """
    if log_format is None:
        log_format = (
            u'[{record.time:%Y-%m-%d %H:%M:%S.%f} pid({record.process})] ' +
            u'{record.level_name}: {record.module}::{record.func_name}:{record.lineno} {record.message}'
        )
    # default_handler = StderrHandler(format_string=log_format)
    default_handler = StreamHandler(sys.stdout, format_string=log_format, level=logbook.INFO)
    default_handler.push_application()
    return LoggerSingle(__name__)



