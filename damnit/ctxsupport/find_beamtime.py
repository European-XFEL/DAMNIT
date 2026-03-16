from typing import Union
from pathlib import Path
from argparse import ArgumentParser

GLOBAL_PATH_P10 = Path('/asap3/petra3/gpfs/p10/')

def find_beamtime(beamtimeId: Union[int, str]):
    """Retrieve path to the P10 beamtime folder."""
    
    if isinstance(beamtimeId, int):
        beamtimeId = str(beamtimeId)
        
    glob_pattern = '*/data/' + beamtimeId

    folders = list()
    for _folder in GLOBAL_PATH_P10.glob(glob_pattern):
        folders.append(_folder)

    if len(folders) == 0:
        print(f"Proposal {proposal} does not exist")
        return None
    elif len(folders) == 1:
        return folders[0].as_posix()
    else:
        raise ValueError(f"multiple entries encountered for proposal {proposal}")


def find_scan(beamtimeId, scan_name):

    beamtime_path = Path(find_beamtime(beamtimeId))

    scan_path = beamtime_path / 'raw' / scan_name

    if scan_path.exists() and scan_path.is_dir():
        return scan_path.as_posix()
    else:
        print(f"Scan {scan_name} does not exist")
        return None        


def main(args=None):
    parser = ArgumentParser(description="Find P10 beamtime/scan directory")
    parser.add_argument('beamtime', help='Beamtime ID, e.g. 12345678')
    parser.add_argument('scan_name', help='Recorded scan name, e.g. m013_ferritin')

    args = parser.parse_args(args)

    print(find_beamtime(args.beamtime))


if __name__ == '__main__':
    main()
        