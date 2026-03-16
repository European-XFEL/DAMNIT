import yaml
import ast
from pathlib import Path
import os

from find_beamtime import find_beamtime, find_scan

def enumerate_scans(beamtime, save_dir=os.getcwd()):
    raw_dir = Path(find_beamtime(beamtime)) / 'raw'

    subdirs = [_dir.name for _dir in raw_dir.iterdir() if _dir.is_dir()]
    subdirs.sort()

    enum_dirs = {i + 1: _name for i, _name in enumerate(subdirs)}

    with open(Path(save_dir) / f"{beamtime}_enumerated_scans.yaml", "w") as f:
        yaml.dump(enum_dirs, f, sort_keys=False)

    print(f"Created enumerated list of scan names for {beamtime}")
    
    
def read_batchinfo(batchfile):

    try:
        with open(batchfile, 'r') as f:
            data = yaml.safe_load(f)

        return data
    except Exception as e:
        print(f"Warning! Exception occurred during reading {str(batchfile)}:")
        print(e)

        return None


def split_by_percent(lines):
    
    result = list()
    _current = [lines[0]]

    for line in lines[1:]:
        if line.startswith("%"):
            # start a new sublist
            result.append(_current.copy())
            _current = [line]
        else:
            _current.append(line)

    # append the last chunk
    result.append(_current)

    return result

def list_to_dict(data):
    
    result = dict()

    for _entry in data:
        _key, _value = _entry.split('=')
        _key = _key.strip()
        _value = _value.strip()
        
        try:
            _value = ast.literal_eval(_value)
        except:
            pass

        result[_key] = _value

    return result


def read_fio(fiofile):

    # reading all lines from the .fio file
    # note: all lines starting with '!' are omitted
    lines = list()
    
    with open(fiofile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("!"):
                continue
                
            lines.append(line)

    multilines = split_by_percent(lines)

    result = dict()

    names = dict()
    names['%c'] = 'command'
    names['%p'] = 'parameters'
    names['%d'] = 'data'

    for _entry in multilines:
        if len(_entry) == 1:
            result[names[_entry[0]]] = None
        else:
            result[names[_entry[0]]] = _entry[1:] 

    if len(result['parameters']) > 0:
        result['parameters'] = list_to_dict(result['parameters'])
    
    return result


def open_scan_via_name(beamtime, scan_name, all_messages=False):

    scan_meta = dict()

    scan_meta['beamtime'] = beamtime
    scan_meta['scan_name'] = scan_name
    
    scan_meta['fio'] = None
    scan_meta['batch'] = None
    
    data_dir = find_scan(beamtime, scan_name)
    scan_meta['scan_path'] = data_dir
    if all_messages:
        print("Data location")
        print(data_dir)

    fiofile = Path(data_dir) / f"{scan_name}.fio"
    if fiofile.exists():
        metafio = read_fio(fiofile)
        scan_meta['fio'] = metafio
    else:
        print("Warning! .fio file does not exist")
        
        return scan_meta

    if 'parameters' in scan_meta['fio'].keys():
        if '_ccd' in scan_meta['fio']['parameters'].keys():
            _ccd = scan_meta['fio']['parameters']['_ccd']
            _ccd_split = _ccd.split(' ')
    else:
        _ccd_split = None

    if _ccd_split is not None:
        if len(_ccd_split) > 1:
            print("Warning! Multiple detector support is not implemented yet")
        else:
            _ccd = _ccd_split[0]

            batchfile = Path(data_dir) / _ccd / f"{scan_name}.batchinfo"
            if batchfile.exists():
                metabatch = read_batchinfo(batchfile)
                scan_meta['batch'] = metabatch
            else:
                print("Warning! .batchinfo file does not exist")
    
    return scan_meta 


def open_scan(beamtime, scan_no, all_messages=False):

    enumerated_list = Path(find_beamtime(beamtime)) / "scratch_cc" / f"{beamtime}_enumerated_scans.yaml"
    
    try:
        with open(enumerated_list, 'r') as f:
            scans = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning! Exception occurred during reading enumerated scan list:")
        print(e)
        return None

    scan_name = scans[scan_no]

    scan_meta = open_scan_via_name(beamtime, scan_name, all_messages=all_messages)
    scan_meta['scan_no'] = scan_no

    return scan_meta
        
    