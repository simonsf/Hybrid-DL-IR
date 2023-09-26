import os
import re
import sys
import importlib
import codecs
from imp import reload


def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist


def read_str_array(file, sep='\t'):
    lines = readlines(file)
    content = []
    for line in lines:
        content.append(re.split(sep, line))
    return content


def load_module_from_disk(pyfile):
    """
    load python module from disk dynamically
    :param pyfile     python file
    :return a loaded network module
    """
    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    modulename, _ = os.path.splitext(basename)

    need_reload = modulename in sys.modules

    # To avoid duplicate module name with existing modules, add the specified path first.
    os.sys.path.insert(0, dirname)
    lib = importlib.import_module(modulename)
    if need_reload:
        reload(lib)
    os.sys.path.pop(0)

    return lib