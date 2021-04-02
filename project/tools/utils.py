import inspect

def get_valid_arguments(cls, params):

    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    return kwargs