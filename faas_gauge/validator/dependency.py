"""
Dependency detector for Python code.

Analyzes Python source code to detect third-party package imports
that need to be installed via pip.
"""

import ast
import sys
import importlib.util
from typing import Set, List, Tuple
from pathlib import Path

# Python standard library modules (Python 3.8+)
# This is a comprehensive list but may need updates for newer Python versions
STDLIB_MODULES = {
    # Built-in modules
    "__future__",
    "__main__",
    "_thread",
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asynchat",
    "asyncio",
    "asyncore",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "cProfile",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "graphlib",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "posixpath",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
    # Commonly used internal modules
    "_abc",
    "_ast",
    "_bisect",
    "_blake2",
    "_codecs",
    "_collections",
    "_collections_abc",
    "_compat_pickle",
    "_compression",
    "_contextvars",
    "_crypt",
    "_csv",
    "_ctypes",
    "_curses",
    "_datetime",
    "_decimal",
    "_elementtree",
    "_frozen_importlib",
    "_functools",
    "_hashlib",
    "_heapq",
    "_imp",
    "_io",
    "_json",
    "_locale",
    "_lsprof",
    "_lzma",
    "_markupbase",
    "_md5",
    "_multibytecodec",
    "_multiprocessing",
    "_opcode",
    "_operator",
    "_osx_support",
    "_pickle",
    "_posixshmem",
    "_posixsubprocess",
    "_py_abc",
    "_pydecimal",
    "_pyio",
    "_queue",
    "_random",
    "_sha1",
    "_sha256",
    "_sha3",
    "_sha512",
    "_signal",
    "_sitebuiltins",
    "_socket",
    "_sqlite3",
    "_sre",
    "_ssl",
    "_stat",
    "_statistics",
    "_string",
    "_strptime",
    "_struct",
    "_symtable",
    "_thread",
    "_threading_local",
    "_tkinter",
    "_tracemalloc",
    "_uuid",
    "_warnings",
    "_weakref",
    "_weakrefset",
    "_xxsubinterpreters",
    "_xxtestfuzz",
    "typing_extensions",
}

# Common package name mappings (pip name -> import name)
IMPORT_TO_PACKAGE = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "google": "google-api-python-client",
}


def extract_imports(code: str) -> Set[str]:
    """
    Extract all import module names from Python code.

    Args:
        code: Python source code.

    Returns:
        Set of top-level module names.
    """
    imports: Set[str] = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module name
                module = alias.name.split(".")[0]
                imports.add(module)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get top-level module name
                module = node.module.split(".")[0]
                imports.add(module)

    return imports


def is_stdlib_module(module_name: str) -> bool:
    """
    Check if a module is part of the Python standard library.

    Args:
        module_name: Name of the module.

    Returns:
        True if module is in stdlib.
    """
    if module_name in STDLIB_MODULES:
        return True

    # Check if it's a built-in module
    if module_name in sys.builtin_module_names:
        return True

    # Try to find the module and check its location
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False

        if spec.origin is None:
            # Built-in module
            return True

        origin = Path(spec.origin)

        # Check if it's in the stdlib directory
        stdlib_path = Path(sys.prefix) / "lib"
        site_packages = "site-packages"

        origin_str = str(origin)
        if site_packages in origin_str:
            return False
        if str(stdlib_path) in origin_str:
            return True

    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    return False


def is_local_module(module_name: str, code_dir: str) -> bool:
    """
    Check if a module is a local file in the same directory.

    Args:
        module_name: Name of the module.
        code_dir: Directory containing the code.

    Returns:
        True if module is a local file.
    """
    code_path = Path(code_dir)

    # Check for .py file
    if (code_path / f"{module_name}.py").exists():
        return True

    # Check for package directory
    if (code_path / module_name / "__init__.py").exists():
        return True

    return False


def get_package_name(import_name: str) -> str:
    """
    Get the pip package name for an import.

    Some packages have different names for pip install vs import.

    Args:
        import_name: The name used in import statement.

    Returns:
        The pip package name.
    """
    return IMPORT_TO_PACKAGE.get(import_name, import_name)


def detect_dependencies(
    code: str, code_dir: str = ".", exclude_local: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Detect third-party dependencies in Python code.

    Args:
        code: Python source code.
        code_dir: Directory containing the code (for local module detection).
        exclude_local: Whether to exclude local modules.

    Returns:
        Tuple of (pip_packages, import_names):
            - pip_packages: List of package names to install with pip
            - import_names: List of original import names
    """
    imports = extract_imports(code)

    third_party_imports = []

    for module in imports:
        # Skip standard library
        if is_stdlib_module(module):
            continue

        # Skip local modules
        if exclude_local and is_local_module(module, code_dir):
            continue

        # Skip SAAF (it will be injected)
        if module == "SAAF":
            continue

        third_party_imports.append(module)

    # Convert to pip package names
    pip_packages = [get_package_name(imp) for imp in third_party_imports]

    return pip_packages, third_party_imports


def check_installed(package: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package: Package name.

    Returns:
        True if installed.
    """
    try:
        importlib.util.find_spec(package.replace("-", "_"))
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def get_missing_packages(packages: List[str]) -> List[str]:
    """
    Get list of packages that are not installed.

    Args:
        packages: List of package names.

    Returns:
        List of missing packages.
    """
    return [pkg for pkg in packages if not check_installed(pkg)]
