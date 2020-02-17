
class UndefinedAlgorithmParameter(Exception):
    def __init__(self, alg, param):
        self.alg = alg
        self.param = param

    def __str__(self):
        msg = "%s parameter is not defined in %s"
        return msg

class UndefinedFeature(Exception):
    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        msg = "%s feature is not defined"
        return msg

class UndefinedExtension(Exception):
    def __init__(self, ext):
        self.ext = ext

    def __str__(self):
        msg = "%s is not target extension"%(self.ext)
        return msg