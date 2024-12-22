import glob
import os
import json
from util.constants import IMAGE_EXTENSIONS

def get_paths(path):
    """
    Get all file paths in the given directory with specified image extensions.

    Args:
        path (str): The directory path to search for files.

    Returns:
        list: A list of file paths with the specified image extensions.
    """
    all_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_paths += glob.glob(os.path.join(path, f"**.{ext}"))
        all_paths += glob.glob(os.path.join(path, f"*/**.{ext}"))
    return all_paths

def can_change_permissions(filepath):
    """
    Check if the current user can change permissions of the given file.

    Args:
        filepath (str): The path to the file.

    Returns:
        bool: True if the user can change permissions, False otherwise.
    """
    euid = os.geteuid()
    file_stat = os.stat(filepath)
    if file_stat.st_uid == euid:
        return True
    if euid == 0:
        return True
    return False

def makedirs(path, permissions=None):
    """
    Create a directory with the specified permissions.

    Args:
        path (str): The directory path to create.
        permissions (int, optional): The permissions to set for the directory. Defaults to 0o777.

    Returns:
        None
    """
    if permissions is not None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if can_change_permissions(path):
                os.chmod(path, permissions)
    else:
        os.makedirs(path, exist_ok=True)

def load_pathfile(pathfile):
    """
    Load a JSON file containing paths and return a dictionary with evaluated keys.

    Args:
        pathfile (str): The path to the JSON file.

    Returns:
        dict: A dictionary with evaluated keys and corresponding values.
    """
    with open(pathfile, "r") as f:
        paths = json.load(f)
    evaled_pathfile = {}
    for k, v in paths.items():
        try:
            evaled_pathfile[int(k)] = v
        except:
            evaled_pathfile[k] = v
    return evaled_pathfile

def load_paths_from_root(root, cls):
    """
    Load all file paths from the root directory for a given class.

    Args:
        root (str): The root directory path.
        cls (str): The class name to search for within the root directory.

    Returns:
        list: A list of file paths for the given class.
    """
    return get_paths(os.path.join(root, cls))