class FunctionBackend(object):
    def __init__(self):
        self.function_classes = {}

    def __getattr__(self, name):
        fn = self.function_classes.get(name)
        if fn is None:
            raise NotImplementedError
        return fn

    def register_function(self, name, function_class):
        self.function_classes[name] = function_class
