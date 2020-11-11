import copy


class ReturnType(object):
    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.extra_repr() + '\n)'

    def extra_repr(self):
        return ''

    def __iter__(self):
        self._value_dict = copy.copy(self.__dict__)
        return self

    def __next__(self):
        if not self._value_dict:
            raise StopIteration
        key = list(self._value_dict.keys())[0]
        value = self._value_dict.pop(key)
        return value


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
