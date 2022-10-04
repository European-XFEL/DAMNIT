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
        '--match', type=str, action="append", default=[],
        help="String to match against variable titles (case-insensitive). Not a regex, simply `str in var.title`."
    )
    reprocess_ap.add_argument(
        'run', nargs='+',
        help="Run number, e.g. 96. Multiple runs can be specified at once, "
             "or pass 'all' to reprocess all runs in the database."
    )

    proposal_ap = subparsers.add_parser(
        'proposal',
        help="Get or set the proposal number to collect metadata from"
    )
    proposal_ap.add_argument(
        'proposal', nargs='?', type=int,
        help="Proposal number to set, e.g. 1234"
    )

    new_id_ap = subparsers.add_parser(
        'new-id',
        help="Set a new (random) database ID. Useful if a copy has been made which should not share an ID with the original."
    )
    new_id_ap.add_argument(
        'db_dir', type=Path, default=Path.cwd(), nargs='?',
        help="Path to the database directory"
    )

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)-38s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.subcmd == 'gui':
        if args.context_dir is not None and not args.context_dir.is_dir():
            sys.exit(f"{args.context_dir} is not a directory")
        from .gui.main_window import run_app
        return run_app(args.context_dir)

    elif args.subcmd == 'listen':
        if args.test:
            from .backend.test_listener import listen
        else:
            from .backend.listener import listen
        os.chdir(args.context_dir)
        return listen()

    elif args.subcmd == 'reprocess':
        # Hide some logging from Kafka to make things more readable
        logging.getLogger('kafka').setLevel(logging.WARNING)

        from .backend.extract_data import Extractor
        extr = Extractor()
        if args.run == ['all']:
            rows = extr.db.execute("SELECT proposal, runnr FROM runs").fetchall()
            print(f"Reprocessing {len(rows)} runs already recorded...")
            for proposal, run in rows:
                extr.extract_and_ingest(proposal, run, match=args.match)
        else:
            try:
                runs = [int(r) for r in args.run]
            except ValueError as e:
                sys.exit(f"Run numbers must be integers ({e})")
            for run in runs:
                extr.extract_and_ingest(args.proposal, run, match=args.match)

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

    elif args.subcmd == 'new-id':
        from secrets import token_hex
        from .backend.db import open_db, set_meta, DB_NAME

        db = open_db(args.db_dir / DB_NAME)
        set_meta(db, "db_id", token_hex(20))



if __name__ == '__main__':
    sys.exit(main())
