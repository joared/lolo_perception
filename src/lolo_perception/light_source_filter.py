import lolo_perception.py_logging as logging
from lolo_perception.light_source import LightSource

class LightSourceFilter:

    def __init__(self, name, activity=None, validity=None):
        self.name = name
        self._activity = activity if activity else {}
        self._validity = validity if validity else {}


    @staticmethod
    def inRange(value, validRange):
        minValue, maxValue = validRange
        if value < minValue or value > maxValue:
            return False
        return True


    def isFilterActive(self, lightSource):
        if self._activity:
            # We have activity defined, check if the light source is in range
            for attribute in self._activity:
                # Get the value of the attribute of the light source
                value = getattr(lightSource, attribute)
                # Get the range of the activity
                valueRange = self._activity[attribute]

                # Check if the value is in range
                if not self.inRange(value, valueRange):
                    # At least one attribute is not in range, this filter is not active
                    return False
                
            # All attributes are in range, this filter is active
            return True
        else:
            # No activity defined, this filter will always be active
            return True


    def isValidLightSource(self, lightSource):
        if self._validity:
            for attribute in self._validity:
                # Get the value of the attribute of the light source
                value = getattr(lightSource, attribute)
                # Get the range of the activity
                valueRange = self._validity[attribute]

                # Check if the value is in range
                if not self.inRange(value, valueRange):
                    # At least one attribute is not in range, this light source is invalid
                    return False
                
            # All attributes are in range, the light source is valid
            return True
        else:
            # No validity defined, reject all light sources
            return False


    def filter(self, lightSource):

        # Check if this filter is active for this light source
        if self.isFilterActive(lightSource):
            # it is active, filter it
            return self.isValidLightSource(lightSource)

        return True
            

    def __repr__(self):
        s = self.name + "\n"
        
        s += "Activity:\n"
        for a in self._activity:
            s += "\t{}: {}\n".format(a, self._activity[a])
        
        s += "Validity:\n"
        for v in self._validity:
            s += "\t{}: {}\n".format(v, self._validity[v])

        return s


class MultiLightSourceFilter:
    def __init__(self, *filters):
        self._filters = []

        for filterSettings in filters:
            name = filterSettings.keys()[0]
            activity = filterSettings[name]["activity"]
            validity = filterSettings[name]["validity"]
            self._filters.append(LightSourceFilter(name, activity, validity))

    def __call__(self, contour, intensity):
        """
        Convenience method, calls filter(lightSource)
        """
        # TODO: filterContour() should be removed and light source should be given in the call
        return self.filter(LightSource(contour, intensity))


    def filter(self, lightSource):
        for f in self._filters:
            if not f.filter(lightSource):
                logging.debug("Light source rejected by filter '{}': {}".format(f.name, lightSource))
                return False
        
        return True


    def __repr__(self):
        s = ""
        for f in self._filters:
            s += str(f)
        
        return s


if __name__ == "__main__":
    pass
