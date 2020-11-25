import pickle
from collections import namedtuple, OrderedDict
from nptorch.tensor import Tensor
from ..parameter import Parameter


def make_indent(s: str):
    s1 = s.split('\n')
    if not s1:
        return s
    s = ['  ' + line for line in s1[1:]]
    s.insert(0, s1[0])
    return '\n'.join(s)


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    def __bool__(self):
        return bool(self.missing_keys or self.unexpected_keys)


class Module(object):
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()

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

    def __setattr__(self, key, value):
        if '.' in key:
            raise RuntimeError('Attribute name of Module cannot contain "."')
        super(Module, self).__setattr__(key, value)

    def extra_repr(self):
        return ''

    def named_children(self):
        children = set()
        for name, value in self.__dict__.items():
            if isinstance(value, Module) and id(value) not in children:
                children.add(id(value))
                yield name, value

    def children(self):
        for name, module in self.named_children():
            yield module

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_parameter(self, name, param):
        self.__setattr__(name, Parameter(param))

    def named_parameters(self, recurse=True):
        params = set()
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter) and id(value) not in params:
                params.add(id(value))
                yield name, value
        if recurse:
            for child in self.children():
                for name, value in child.named_parameters():
                    if id(value) not in params:
                        params.add(id(value))
                        yield name, value

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse):
            yield param

    def requires_grad_(self, mode=True):
        for p in self.parameters():
            p.requires_grad_(mode)
        return self

    def apply(self, fcn):
        for module in self.children():
            module.apply(fcn)
        fcn(self)
        return self

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def state_dict(self):
        state_dict = OrderedDict({k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)})
        for name, module in self.named_children():
            state_dict.update(OrderedDict({f'{name}.{k}': v for k, v in module.state_dict().items()}))
        return state_dict

    def save_state_dict(self, state_dict_file):
        state_dict = self.state_dict()
        with open(state_dict_file, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self, state_dict: OrderedDict or str, strict=True):
        """
        载入参数
        @param state_dict: 参数字典或字典文件路径
        @param strict: 字典的键是否需要完全匹配
        """
        if isinstance(state_dict, str):
            with open(state_dict, 'rb') as f:
                state_dict = pickle.load(f)
        if not isinstance(state_dict, OrderedDict):
            raise TypeError(f'state_dict must be type: `OrderedDict`, Got {type(state_dict)}')
        model_state_dict_keys = self.state_dict().keys()
        state_dict_keys = state_dict.keys()
        missing_keys = model_state_dict_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_state_dict_keys
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if strict and incompatible_keys:
            error_msg = f'Error(s) in loading state_dict for {self.__class__.__name__}:'
            if missing_keys:
                missing_keys = '"' + '", "'.join(missing_keys) + '"'
                error_msg += f'\nMissing keys in state_dict: {missing_keys}'
            if unexpected_keys:
                unexpected_keys = '"' + '", "'.join(unexpected_keys) + '"'
                error_msg += f'\nUnexpected keys in state_dict: {unexpected_keys}'
            raise RuntimeError(error_msg)
        for key, value in state_dict.items():
            keys = key.split('.')
            module = self
            for k in keys[:-1]:
                if not hasattr(module, k):
                    break
                module = getattr(module, k)
            else:
                if hasattr(module, keys[-1]):
                    module.__setattr__(keys[-1], value)
        return incompatible_keys

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
