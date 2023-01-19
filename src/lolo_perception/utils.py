import time

class Timer:

    class TimerError(Exception):
        #https://realpython.com/python-timer/#a-python-timer-class
        """A custom exception used to report errors in use of Timer class"""

    def __init__(self, name, nAveragSamples=1):
        self.name = name
        self._startTime = None
        self._endTime = None

        self.nAveragSamples = nAveragSamples
        self._times = []
        #self._avg = 0
        #self._n = 0

    def reset(self):
        self._startTime = None
        self._endTime = None

    def start(self):
        if self._startTime is not None:
            self.TimerError("Timer '{}' is already started, use stop() to stop timer.".format(self.name))

        self._startTime = time.time()

    def stop(self):
        if self._startTime:
            self._endTime = time.time()
        else:
            raise self.TimerError("Timer '{}' is not started, use start() to start timer.".format(self.name))

        #self._avg = (self._avg*self._n + self.elapsed())/float(self._n+1)
        #self._n += 1
        if self.nAveragSamples > 1:
            self._times.append(self.elapsed())
            self._times = self._times[-self.nAveragSamples:]

    def elapsed(self):
        if self._startTime and self._endTime:
            return self._endTime - self._startTime
        else:
            return 0

    def avg(self):
        if self.nAveragSamples > 1:
            if len(self._times) ==  0:
                raise self.TimerError("Timer '{}' has not recorded any times yet. Use start() and stop() to record.".format(self.name))
            return sum(self._times)/float(len(self._times))
        else:
            return self.elapsed()