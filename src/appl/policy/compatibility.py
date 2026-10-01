"""Resolve public capability imports in byte-preserved API submissions.

Generated policy files retain their original bytes. Only these two public numeric
capability names are aliases; historical experiment runners are not imported.
"""
import importlib
import sys
import types


def install():
    public = importlib.import_module('appl.policy.public')
    for name in ('experiments.exp2.sol_flexible.public', 'experiments.exp2.six_tasks.flexible.public'):
        parts = name.split('.')
        for count in range(1, len(parts)):
            parent = '.'.join(parts[:count])
            if parent not in sys.modules:
                module = types.ModuleType(parent)
                module.__path__ = []
                sys.modules[parent] = module
            if count > 1:
                setattr(sys.modules['.'.join(parts[:count-1])], parts[count-1], sys.modules[parent])
        sys.modules[name] = public
        setattr(sys.modules[name.rsplit('.', 1)[0]], 'public', public)
