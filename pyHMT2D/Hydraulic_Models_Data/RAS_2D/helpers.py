"""

Some of the following code is adapted from pyras under the MIT license: https://github.com/solomonvimal/pyras

"""

import os

import win32api
import win32con

from pyHMT2D.Misc import yes_or_no


def get_supported_hec_ras_versions():
    """Return a list of supported HEC-RAS versions
    """
    return ['5.0.7', '6.0.1']


def kill_all_hec_ras():
    """ Kill all running HEC-RAS instances

    Close all HEC-RAS instances. This is potentially dangerous because all running HEC-RAS processes will be killed
    without warning. Make sure you save and close all HEC-RAS instances before you call this.

    """
    import os
    import subprocess

    print("Kill all HEC-RAS instances. Proceed with caution.")

    if (not yes_or_no("Are you sure?")):
        print("Ok, no HEC-RAS instance will be killed.")
        return

    ras_process_string = b'ras.exe'
    proc = subprocess.Popen('TASKLIST /FO "CSV"', stdout=subprocess.PIPE)
    #print("proc.stdout.read() = ", proc.stdout.read())
    tasklist = proc.stdout.read().split(b'\n')
    tasks = []
    pids = []
    for line in tasklist:
        l = line.lower()
        if ras_process_string in l:
            items = l.split(b',')
            tasks.append(items)
            pids.append(int(eval(items[1])))

    print("Found ", len(pids), "running HEC-RAS instances. Kill them all.")

    for pid in pids:
        try:
            os.system('TASKKILL /PID {0} /F >nul'.format(pid))
        except Exception as e:
            print(e)


def get_installed_hec_ras_versions():
    """ Get a list of installed HEC-RAS versions
    """
    #this list has to include as many possible HEC-RAS versions as possible
    ver = {'HEC-RAS\\6.0 Beta Update 2\\Ras.exe': '6.0.1',
           'HEC-RAS\\5.0.7\\Ras.exe': '5.0.7'}

    ldic = _get_registered_typelibs()

    # Check if files actually exist (another sanity check)
    available_versions = []

    for dic in ldic:
        fname = dic['filename']
        if os.path.isfile(fname):
            for k in ver:
                if k in fname:
                    available_versions.append(ver[k])
    return available_versions


def _get_typelib_info(keyid, version):
    """
    adapted from pywin32

    # Copyright (c) 1996-2008, Greg Stein and Mark Hammond.
    """
    collected = []
    help_path = ""
    key = win32api.RegOpenKey(win32con.HKEY_CLASSES_ROOT,
                              "TypeLib\\%s\\%s" % (keyid, version))
    try:
        num = 0
        while 1:
            try:
                sub_key = win32api.RegEnumKey(key, num)
            except win32api.error:
                break
            h_sub_key = win32api.RegOpenKey(key, sub_key)
            try:
                value, typ = win32api.RegQueryValueEx(h_sub_key, None)
                if typ == win32con.REG_EXPAND_SZ:
                    value = win32api.ExpandEnvironmentStrings(value)
            except win32api.error:
                value = ""
            if sub_key == "HELPDIR":
                help_path = value
            elif sub_key == "Flags":
                flags = value
            else:
                try:
                    lcid = int(sub_key)
                    lcidkey = win32api.RegOpenKey(key, sub_key)
                    # Enumerate the platforms
                    lcidnum = 0
                    while 1:
                        try:
                            platform = win32api.RegEnumKey(lcidkey, lcidnum)
                        except win32api.error:
                            break
                        try:
                            hplatform = win32api.RegOpenKey(lcidkey, platform)
                            fname, typ = win32api.RegQueryValueEx(hplatform, None)
                            if typ == win32con.REG_EXPAND_SZ:
                                fname = win32api.ExpandEnvironmentStrings(fname)
                        except win32api.error:
                            fname = ""
                        collected.append((lcid, platform, fname))
                        lcidnum = lcidnum + 1
                    win32api.RegCloseKey(lcidkey)
                except ValueError:
                    pass
            num = num + 1
    finally:
        win32api.RegCloseKey(key)

    return fname, lcid


def _get_registered_typelibs(match='HEC River Analysis System'):
    """
    adapted from pywin32

    # Copyright (c) 1996-2008, Greg Stein and Mark Hammond.
    """
    # Explicit lookup in the registry.
    result = []
    key = win32api.RegOpenKey(win32con.HKEY_CLASSES_ROOT, "TypeLib")
    try:
        num = 0
        while 1:
            try:
                key_name = win32api.RegEnumKey(key, num)
            except win32api.error:
                break
            # Enumerate all version info
            sub_key = win32api.RegOpenKey(key, key_name)
            name = None
            try:
                sub_num = 0
                best_version = 0.0
                while 1:
                    try:
                        version_str = win32api.RegEnumKey(sub_key, sub_num)
                    except win32api.error:
                        break
                    try:
                        version_flt = float(version_str)
                    except ValueError:
                        version_flt = 0  # ????
                    if version_flt > best_version:
                        best_version = version_flt
                        name = win32api.RegQueryValue(sub_key, version_str)
                    sub_num = sub_num + 1
            finally:
                win32api.RegCloseKey(sub_key)
            if name is not None and match in name:
                fname, lcid = _get_typelib_info(key_name, version_str)

                # Split version
                major, minor = version_str.split('.')

                result.append({'name': name,
                               'filename': fname,
                               'iid': key_name,
                               'lcid': lcid,
                               'major': int(major),
                               'minor': int(minor)})
            num = num + 1
    finally:
        win32api.RegCloseKey(key)

    #print(result)
    #result = sorted(result)

    return result

