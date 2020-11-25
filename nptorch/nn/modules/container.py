from .module import Module


class Container(Module):
    def __init__(self):
        super(Container, self).__init__()

    def __len__(self):
        return len(list(self.children()))

    def forward(self, *args):
        raise NotImplementedError


class Sequential(Container):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def __getitem__(self, item):
        keys = list(self.__dict__.get('_modules'))
        return self._modules[keys[item]]

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class ModuleList(Container):
    def __init__(self, modules):
        super(ModuleList, self).__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __getitem__(self, item):
        keys = list(self.__dict__.get('_modules'))
        return self._modules[keys[item]]

    def append(self, module):
        self.add_module(str(self.__len__()), module)

    def extend(self, modules):
        for i, module in enumerate(modules, self.__len__()):
            self.add_module(str(i), module)

    def forward(self, *args):
        raise NotImplementedError
