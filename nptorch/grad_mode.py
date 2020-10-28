class no_grad(object):
    def __enter__(self):
        grad_enable.set_grad_enable(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        grad_enable.set_grad_enable(True)


class is_grad_enable(object):
    def __init__(self):
        self._grad_enable = True

    def __call__(self):
        return self._grad_enable

    def set_grad_enable(self, mode):
        self._grad_enable = mode


grad_enable = is_grad_enable()
