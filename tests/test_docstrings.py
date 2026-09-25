import importlib
import inspect
import pkgutil
import re

import pytest

import rungrid


def get_all_submodules(package):
    submodules = []
    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        submodules.append(importlib.import_module(module_name))
    return submodules


@pytest.mark.parametrize("module", [rungrid] + get_all_submodules(rungrid))
def test_all_public_docstrings(module):
    # Check module docstring
    assert module.__doc__ is not None, f"Module {module.__name__} has no docstring"
    assert len(module.__doc__.strip()) > 0, (
        f"Module {module.__name__} has an empty docstring"
    )

    # Walk all public items in the module
    for name, obj in inspect.getmembers(module):
        # We only care about public items or dunder methods
        is_private = name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )
        if is_private:
            continue

        # Skip imports and builtins
        if inspect.ismodule(obj):
            continue
        obj_module = getattr(obj, "__module__", None)
        if obj_module and not obj_module.startswith("rungrid"):
            continue

        if inspect.isclass(obj):
            # Check class docstring
            assert obj.__doc__ is not None, (
                f"Class {obj.__module__}.{obj.__name__} has no docstring"
            )
            assert len(obj.__doc__.strip()) > 0, (
                f"Class {obj.__module__}.{obj.__name__} has an empty docstring"
            )

            # Get class source code to check for explicit declarations
            try:
                class_source = inspect.getsource(obj)
            except (TypeError, OSError):
                class_source = ""

            # Check methods
            for method_name, method in inspect.getmembers(obj):
                # Skip private methods
                method_private = method_name == "__repr__" or (
                    method_name.startswith("_")
                    and not (
                        method_name.startswith("__") and method_name.endswith("__")
                    )
                )
                if method_private:
                    continue

                # Ensure it's defined directly in this class (not inherited)
                if method_name not in obj.__dict__:
                    continue

                # Verify that this method is explicitly defined in the class source code
                # (avoids checking auto-generated dataclass/typing/Protocol methods)
                if class_source:
                    has_explicit_def = re.search(
                        r"\bdef\s+" + re.escape(method_name) + r"\b", class_source
                    )
                    if not has_explicit_def:
                        continue

                method_obj = getattr(obj, method_name)
                if not (
                    inspect.isfunction(method_obj)
                    or inspect.ismethod(method_obj)
                    or isinstance(method_obj, (property, classmethod, staticmethod))
                ):
                    continue

                doc = None
                if isinstance(method_obj, property):
                    doc = method_obj.__doc__
                elif isinstance(method_obj, (classmethod, staticmethod)):
                    doc = method_obj.__get__(None, obj).__doc__
                else:
                    doc = method_obj.__doc__

                assert doc is not None, (
                    f"Method {obj.__module__}.{obj.__name__}.{method_name} has no docstring"
                )
                assert len(doc.strip()) > 0, (
                    f"Method {obj.__module__}.{obj.__name__}.{method_name} has an empty docstring"
                )

        elif inspect.isfunction(obj):
            # Check function docstring
            assert obj.__doc__ is not None, (
                f"Function {obj.__module__}.{obj.__name__} has no docstring"
            )
            assert len(obj.__doc__.strip()) > 0, (
                f"Function {obj.__module__}.{obj.__name__} has an empty docstring"
            )
