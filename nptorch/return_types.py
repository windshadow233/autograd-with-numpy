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

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter > 1:
            raise StopIteration
        if self.counter == 0:
            self.counter += 1
            return self.values
        self.counter += 1
        return self.indices


class Min(ReturnType):
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def extra_repr(self):
        return f'\tvalues={self.values}\n\tindices={self.indices}'

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter > 1:
            raise StopIteration
        if self.counter == 0:
            self.counter += 1
            return self.values
        self.counter += 1
        return self.indices


class Sort(ReturnType):
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def extra_repr(self):
        return f'\tvalues={self.values}\n\tindices={self.indices}'

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter > 1:
            raise StopIteration
        if self.counter == 0:
            self.counter += 1
            return self.values
        self.counter += 1
        return self.indices
