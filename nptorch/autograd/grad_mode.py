import functools


class no_grad(object):
    def __enter__(self):
        grad_enable.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        grad_enable.set_grad_enabled(True)

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_no_grad


class GradEnable(object):
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super(GradEnable, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._grad_enable = True

    def __call__(self):
        return self._grad_enable

    def set_grad_enabled(self, mode):
        self._grad_enable = mode


grad_enable = GradEnable()

