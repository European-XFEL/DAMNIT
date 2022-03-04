class Variable:
    def __init__(self, summary=None):
        self.func = None
        self.summary = summary

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self
