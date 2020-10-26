import pickle
from ..parameter import Parameter, Parameters


def make_indent(s: str):
    s1 = s.split('\n')
    if not s1:
        return s
    s = ['  ' + line for line in s1[1:]]
    s.insert(0, s1[0])
    return '\n'.join(s)


class Module(object):
    def __init__(self):
        self.training = True

    def __repr__(self):
        extra_repr = self.extra_repr()
        extra_lines = extra_repr.split('\n') if extra_repr else []
        child_lines = []
        for name, child in self.named_children():
            mod_str = make_indent(repr(child))
            child_lines.append('(' + name + '): ' + mod_str)
        lines = extra_lines + child_lines
        s = self.__class__.__name__ + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                s += extra_lines[0]
            else:
                s += '\n  ' + '\n  '.join(lines) + '\n'
        s += ')'
        return s

    def __call__(self, *args):
        return self.forward(*args)

    def extra_repr(self):
        return ''

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self, is_first_call=True):
        params = [v for v in self.__dict__.values() if isinstance(v, Parameter)]
        for module in self.children():
            params.extend(module.parameters(False))
        if is_first_call:
            return Parameters(params)
        else:
            return params

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def forward(self, *args):
        raise NotImplementedError

