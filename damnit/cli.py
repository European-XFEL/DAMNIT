import inspect
import logging
import os
import sys
import textwrap
import traceback
from argparse import ArgumentParser
from pathlib import Path

from termcolor import colored

from extra_data.read_machinery import find_proposal


def excepthook(exc_type, value, tb):
    """
    Hook to start an IPython shell when an unhandled exception is caught.
    """
    tb_msg = "".join(traceback.format_exception(exc_type, value, tb))
    traceback.print_exception(exc_type, value, tb)
    print()

    # Find the deepest frame in the stack that comes from us. We don't want to
    # go straight to the last frame because that may be in some other library.
    module_path = Path(__file__).parent.parent
    target_frame = None
    target_file = None
    for frame, lineno in traceback.walk_tb(tb):
        frame_file = Path(inspect.getframeinfo(frame).filename)
        if module_path in frame_file.parents:
            target_frame = frame
            target_file = frame_file.relative_to(module_path)

    # Start an IPython REPL
    header = f"""
    Tip: call {colored('__tb()', 'red')} to print the traceback again.
    Dropped into {colored(target_file, 'green')} at line {colored(target_frame.f_lineno, 'green')}.
    """
    print(textwrap.dedent(header))

    import IPython
    IPython.start_ipython(argv=[], display_banner=False,
                          user_ns=target_frame.f_locals | target_frame.f_globals | {"__tb": lambda: print(tb_msg)})


def main():
    ap = ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="Show debug logs.")
    ap.add_argument('--debug-repl', action='store_true',
                    help="Drop into an IPython repl if an exception occurs. Local variables at the point of the exception will be available.")
    subparsers = ap.add_subparsers(required=True, dest='subcmd')

    gui_ap = subparsers.add_parser('gui', help="Launch application")
    gui_ap.add_argument(
        'proposal_or_dir', nargs='?',
        help="Either a proposal number or a database directory."
    )
    gui_ap.add_argument(
        '--no-kafka', action='store_true',
        help="Don't try connecting to XFEL's Kafka broker"
    )

    listen_ap = subparsers.add_parser(
        'listen', help="Watch for new runs & extract data from them"
    )
    listen_args_grp = listen_ap.add_mutually_exclusive_group()
    listen_args_grp.add_argument(
        '--test', action='store_true',
        help="Manually enter 'migrated' runs for testing"
    )
    listen_args_grp.add_argument(
        '--daemonize', action='store_true',
        help="Start the listener under a separate process managed by supervisord."
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
        "--mock", action="store_true",
        help="Use a fake run object instead of loading one from disk."
             " Note: do not use the passed `run` object in your context file with this"
             " flag enabled, it will not contain any useful data."
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

    readctx_ap = subparsers.add_parser(
        'read-context',
        help="Re-read the context file and update variables in the database"
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

    config_ap = subparsers.add_parser(
        'db-config',
        help="See or change config in this database"
    )
    config_ap.add_argument(
        '-d', '--delete', action='store_true',
        help="Delete the specified key",
    )
    config_ap.add_argument(
        'key', nargs='?',
        help="The config key to see/change. If not given, list all config"
    )
    config_ap.add_argument(
        'value', nargs='?',
        help="A new value for the given key"
    )

    migrate_ap = subparsers.add_parser(
        "migrate",
        help="Execute migrations to help upgrading. Do NOT execute a migration unless you know what you're doing."
    )
    migrate_ap.add_argument(
        "--dry-run", action="store_true"
    )
    migrate_subparsers = migrate_ap.add_subparsers(dest="migrate_subcmd")
    migrate_subparsers.add_parser(
        "v0-to-v1",
        help="Migrate the SQLite database and HDF5 files from v0 to v1."
    )
    migrate_subparsers.add_parser(
        "intermediate-v1",
        help="Migrate the SQLite database HDF5 files from an initial implementation of v1 to the final"
             " v1. Don't use this unless you know what you're doing."
    )

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)-38s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.debug_repl:
        sys.excepthook = excepthook

    if args.subcmd == 'gui':
        if args.proposal_or_dir is not None:
            if (path := Path(args.proposal_or_dir)).is_dir():
                context_dir = path
            elif args.proposal_or_dir.isdigit():
                proposal_name = f"p{int(args.proposal_or_dir):06d}"
                context_dir = Path(find_proposal(proposal_name)) / "usr/Shared/amore"
            else:
                sys.exit(f"{args.proposal_or_dir} is not a proposal number or DAMNIT database directory")
        else:
            context_dir = None

        from .gui.main_window import run_app
        return run_app(context_dir, connect_to_kafka=not args.no_kafka)

    elif args.subcmd == 'listen':
        from .backend.db import db_path
        from .backend import initialize_and_start_backend

        if args.daemonize:
            if not db_path(args.context_dir).is_file():
                sys.exit("You must create a database with `amore-proto proposal` before starting the listener.")

            return initialize_and_start_backend(args.context_dir)
        else:
            if args.test:
                from .backend.test_listener import listen
            else:
                from .backend.listener import listen

            os.chdir(args.context_dir)
            return listen()

    elif args.subcmd == 'reprocess':
        # Hide some logging from Kafka to make things more readable
        logging.getLogger('kafka').setLevel(logging.WARNING)

        from .backend.extract_data import reprocess
        reprocess(args.run, args.proposal, args.match, args.mock)

    elif args.subcmd == 'read-context':
        from .backend.extract_data import Extractor
        Extractor().update_db_vars()

    elif args.subcmd == 'proposal':
        from .backend.db import DamnitDB
        db = DamnitDB()
        currently_set = db.metameta.get('proposal', None)
        if args.proposal is None:
            print("Current proposal number:", currently_set)
        elif args.proposal == currently_set:
            print(f"No change - proposal {currently_set} already set")
        else:
            db.metameta['proposal'] = args.proposal
            print(f"Changed proposal to {args.proposal} (was {currently_set})")

    elif args.subcmd == 'new-id':
        from secrets import token_hex
        from .backend.db import DamnitDB

        db = DamnitDB.from_dir(args.db_dir)
        db.metameta["db_id"] = token_hex(20)

    elif args.subcmd == 'db-config':
        from .backend.db import DamnitDB

        if args.key:
            args.key = args.key.replace('-', '_')

        db = DamnitDB()
        if args.delete:
            if not args.key:
                sys.exit("Error: no key specified to delete")
            del db.metameta[args.key]
        elif args.key and (args.value is not None):
            db.metameta[args.key] = args.value
        elif args.key:
            try:
                print(repr(db.metameta[args.key]))
            except KeyError:
                sys.exit(f"Error: key {args.key} not found")
        else:
            for k, v in db.metameta.items():
                print(f"{k}={v!r}")

    elif args.subcmd == "migrate":
        from .backend.db import DamnitDB
        from .migrations import migrate_intermediate_v1, migrate_v0_to_v1

        db = DamnitDB(allow_old=True)

        if args.migrate_subcmd == "v0-to-v1":
            migrate_v0_to_v1(db, Path.cwd(), args.dry_run)
        elif args.migrate_subcmd == "intermediate-v1":
            migrate_intermediate_v1(db, Path.cwd(), args.dry_run)

if __name__ == '__main__':
    sys.exit(main())
