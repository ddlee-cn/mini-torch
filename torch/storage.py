import torch
from ._utils import _type

class _StorageBase(object):
    is_cuda = False
    is_sparse = False

    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in _range(len(self)))
        return content + '\n[{} of size {}]'.format(torch.typename(self), len(self))

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], _range(self.size())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        return type(self), (self.tolist(),)

    def clone(self):
        """Returns a copy of this storage"""
        return type(self)(self.size()).copy_(self)

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return [v for v in self]

    def float(self):
        """Casts this storage to float type"""
        return self.type(type(self).__module__ + '.FloatStorage')

_StorageBase.type = _type
