class ReturnType(object):
    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.extra_repr() + '\n)'

    def extra_repr(self):
        return ''


class Max(ReturnType):
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def extra_repr(self):
        return f'\tvalues={self.values}\n\tindices={self.indices}'


class Min(ReturnType):
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def extra_repr(self):
        return f'\tvalues={self.values}\n\tindices={self.indices}'


class Sort(ReturnType):
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def extra_repr(self):
        return f'\tvalues={self.values}\n\tindices={self.indices}'
