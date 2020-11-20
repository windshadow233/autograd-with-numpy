import pickle
from collections import defaultdict, OrderedDict
from nptorch.tensor import Tensor
from ..parameter import Parameter


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

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def extra_repr(self):
        return ''

    def named_children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value

    def children(self):
        for name, module in self.named_children():
            yield module

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        self.train(False)

    def register_parameter(self, name, param):
        self.__setattr__(name, Parameter(param))

    def named_parameters(self, recurse=True):
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                yield name, value
        if recurse:
            for child in self.children():
                for name, value in child.named_parameters():
                    yield name, value

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse):
            yield param

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def state_dict(self):
        state_dict = OrderedDict({k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)})
        for name, module in self.named_children():
            child_state_dict = module.state_dict()
            state_dict.update(OrderedDict({f'{name}.{k}': v for k, v in child_state_dict.items()}))
        return state_dict

    def save_state_dict(self, state_dict_file_name):
        state_dict = self.state_dict()
        with open(state_dict_file_name, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self, state_dict: OrderedDict or str):
        """
        载入参数
        @param state_dict: 参数字典或字典文件路径
        """
        if isinstance(state_dict, str):
            with open(state_dict, 'rb') as f:
                state_dict = pickle.load(f)
        if not state_dict:
            return
        child_state_dict = defaultdict(OrderedDict)
        for key, value in state_dict.items():
            if key.count('.') == 0:
                if not hasattr(self, key):
                    raise AttributeError(f'{self.__class__.__name__} object has no attribute `{key}`')
                self.__setattr__(key, value)
            else:
                child_name, key = key.split('.', 1)
                if not hasattr(self, child_name):
                    raise AttributeError(f'{self.__class__.__name__} object has no attribute `{child_name}`')
                child_state_dict[child_name][key] = value
        for name, module in self.named_children():
            module.load_state_dict(child_state_dict.get(name))

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

