#from shared_logger import get_logger
# 共有したいTimelineLoggerのインスタンスを保持する

_logger_instance = None

def set_logger(logger_instance):
    global _logger_instance
    _logger_instance = logger_instance

def get_logger():
    return _logger_instance