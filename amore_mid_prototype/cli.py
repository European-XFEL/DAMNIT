import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

def main():
    ap = ArgumentParser()
    ap.add_argument('--debug', action='store_true')
    subparsers = ap.add_subparsers(required=True, dest='subcmd')

    gui_ap = subparsers.add_parser('gui', help="Launch application")
    gui_ap.add_argument(
        'context_dir', type=Path, nargs='?',
        help="Directory storing summarised results"
    )

    listen_ap = subparsers.add_parser(
        'listen', help="Watch for new runs & extract data from them"
    )
    listen_ap.add_argument(
        '--test', action='store_true',
        help="Manually enter 'migrated' runs for testing"
    )
    listen_ap.add_argument(
        'context_dir', type=Path, nargs='?', default='.',
        help="Directory to store summarised results"
    )

    reprocess_ap = subparsers.add_parser(
        'reprocess',
        help="Extract data from specified runs. This does not send live updates yet."
    )
    reprocess_ap.add_argument(
        '--proposal', type=int,
        help="Proposal number, e.g. 1234"
    )
    reprocess_ap.add_argument(
        'run', nargs='+', type=int,
        help="Run number, e.g. 96. Multiple runs can be specified at once."
    )

    proposal_ap = subparsers.add_parser(
        'proposal',
        help="Get or set the proposal number to collect metadata from"
    )
    proposal_ap.add_argument(
        'proposal', nargs='?', type=int,
        help="Proposal number to set, e.g. 1234"
    )

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.subcmd == 'gui':
        if args.context_dir is not None and not args.context_dir.is_dir():
            sys.exit(f"{args.context_dir} is not a directory")
        from .gui.main_window import run_app
        return run_app(args.context_dir)

    elif args.subcmd == 'listen':
        if args.test:
            from .backend.test_listener import listen_migrated
        else:
            from .backend.listener import listen_migrated
        os.chdir(args.context_dir)
        return listen_migrated()

    elif args.subcmd == 'reprocess':
        from .backend.extract_data import extract_and_ingest
        for run in args.run:
            extract_and_ingest(args.proposal, run)

    elif args.subcmd == 'proposal':
        from .backend.db import open_db, get_meta, set_meta
        db = open_db()
        currently_set = get_meta(db, 'proposal', None)
        if args.proposal is None:
            print("Current proposal number:", currently_set)
        elif args.proposal == currently_set:
            print(f"No change - proposal {currently_set} already set")
        else:
            set_meta(db, 'proposal', args.proposal)
            print(f"Changed proposal to {args.proposal} (was {currently_set})")



if __name__ == '__main__':
    sys.exit(main())
