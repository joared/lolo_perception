import sys
import os
import datetime

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
TRACE = 5
NOTSET = 0

_levelToName = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    TRACE: 'TRACE',
    NOTSET: 'NOTSET',
}
_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'TRACE': TRACE,
    'NOTSET': NOTSET,
}

def dummy():
    pass

# dummy is arbitrary, Could be any local code
_srcfile = os.path.normcase(dummy.__code__.co_filename)
currentframe = lambda: sys._getframe(1)

def _isInternalFrame(frame):
    """Signal whether the frame is a CPython or logging module internal."""
    filename = os.path.normcase(frame.f_code.co_filename)
    return filename == _srcfile or (
        "importlib" in filename and "_bootstrap" in filename
    )

class Logger:
    def __init__(self, 
                 name="Logger", 
                 level=NOTSET, 
                 printLevel=WARNING, 
                 format="[{levelname:^8s}]:[{timestamp}]:[{file:^20s}]:[{funcname: ^15s}]:[{lineno:^4}]: {message}", 
                 printFormat="[{levelname:^8s}]:[{messageindex:0>4}]: {message}",
                 filename=None, 
                 silent=False):
        self.name = name
        self.level = level
        self.printLevel = printLevel if printLevel else level
        self.msgFormat = format
        self.printFormat = printFormat
        self.filename = filename
        self.silent = silent

        # Keep track of the number of logged messages
        self._msgIndex = 0

    def critical(self, msg, *args, **kwargs):
        self._log(CRITICAL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log(ERROR, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._log(WARNING, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log(DEBUG, msg, *args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        self._log(TRACE, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self._log(level, msg, *args, **kwargs)

    def findCaller(self, stackLevel=1):
        frame = currentframe()
        while stackLevel > 0:
            next_frame = frame.f_back
            if next_frame is None:
                ## We've got options here.
                ## If we want to use the last (deepest) frame:
                break
                ## If we want to mimic the warnings module:
                #return ("sys", 1, "(unknown function)", None)
                ## If we want to be pedantic:
                #raise ValueError("call stack is not deep enough")
            frame = next_frame
            if not _isInternalFrame(frame):
                stackLevel -= 1
        co = frame.f_code
        return co.co_filename, frame.f_lineno, co.co_name

    def _log(self, level, msg, stackLevel=1):
        if level >= self.level or level >= self.printLevel:
            filename, lineno, funcName = self.findCaller(stackLevel)
            filename = os.path.basename(filename)

            if level >= self.level:
                if self.filename:
                    s = self.msgFormat.format(levelname=_levelToName[level], 
                                      timestamp=datetime.datetime.today(),
                                      messageindex=self._msgIndex,
                                      file=filename, 
                                      lineno=lineno, 
                                      funcname=funcName, 
                                      message=msg)
                    with open(self.filename, "a") as f:
                        f.write(s + "\n")
            
            if not self.silent and level >= self.printLevel:
                s = self.printFormat.format(levelname=_levelToName[level], 
                                            timestamp=datetime.datetime.today(),
                                            messageindex=self._msgIndex,
                                            file=filename, 
                                            lineno=lineno, 
                                            funcname=funcName, 
                                            message=msg)
                print(s)

            # Increase the message index.
            self._msgIndex += 1


_defaultLoggerInstance = Logger("Default logger")

def basicConfig(**kwargs):
    global _defaultLoggerInstance
    _defaultLoggerInstance = Logger(**kwargs)

def critical(msg, *args, **kwargs):
    _defaultLoggerInstance.critical(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    _defaultLoggerInstance.error(msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
    _defaultLoggerInstance.warn(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    _defaultLoggerInstance.info(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    _defaultLoggerInstance.debug(msg, *args, **kwargs)

def trace(msg, *args, **kwargs):
    _defaultLoggerInstance.trace(msg, *args, **kwargs)

def log(level, msg):
    _defaultLoggerInstance.log(level, msg)

if __name__ == "__main__":
    l = Logger(name="My Logger", level=DEBUG, fileName="../../logging/my_log.log")
    
    def hey():
        for i in range(10):
            debug("Message debug {}".format(i+1), stackLevel=1)

    hey()

    