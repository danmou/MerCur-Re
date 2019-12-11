# __init__.py
#
# (C) 2019, Daniel Mouritzen

from .general import get_distribution_strategy, tf_nested_py_func, trace_graph

__all__ = ['get_distribution_strategy', 'tf_nested_py_func', 'trace_graph']
