import sys
import importlib

def reload_all():
    """
    Reload all modules inside the src package.
    Deepest submodules are reloaded first, then parents, then src itself.
    """
    modules = [name for name in sys.modules if name == "src" or name.startswith("src.")]
    # Sort by depth (deepest first), so e.g. src.models.layers before src.models
    for name in sorted(modules, key=lambda n: n.count("."), reverse=True):
        importlib.reload(sys.modules[name])