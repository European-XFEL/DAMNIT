import inspect
import logging
import os
import sys
import textwrap
import traceback
from argparse import ArgumentParser
from pathlib import Path

from termcolor import colored
from IPython.terminal.embed import InteractiveShellEmbed

from extra_data.read_machinery import find_proposal


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

    try:
        if args.subcmd == 'gui':
            if args.proposal_or_dir is not None:
                if (path := Path(args.proposal_or_dir)).is_dir():
                    context_dir = path
                elif args.proposal_or_dir.isdigit():
                    proposal_name = f"p{int(args.proposal_or_dir):06d}"
                    context_dir = Path(find_proposal(proposal_name)) / "usr/Shared/amore"
                else:
                    sys.exit(f"{args.proposal_or_dir} is not a proposal number or directory")
            else:
                context_dir = None

            from .gui.main_window import run_app
            return run_app(context_dir, connect_to_kafka=not args.no_kafka)

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
    except Exception as e:
        # If we're going to drop into a REPL, save the current traceback for
        # later. We need to run the REPL outside of the except clause because
        # otherwise the caught exception will stay on the exception stack and be
        # reported along with any exceptions from the REPL itself. e.g. you
        # misspell a variable in the REPL and get the entire traceback followed
        # by a couple of lines for the NameError.
        if args.debug_repl:
            exc_type, value, tb = sys.exc_info()
            tb_msg = traceback.format_exc()
            traceback.print_exc()
            print()
        else:
            raise e from None
    else:
        return 0

    ## This code should never be reached unless an exception was thrown and a
    ## REPL requested.

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
    repl = InteractiveShellEmbed()
    header = f"""
    Tip: call {colored('__tb()', 'red')} to print the traceback again.
    Dropped into {colored(target_file, 'green')} at line {colored(target_frame.f_lineno, 'green')}.
    """
    repl(header=textwrap.dedent(header),
         local_ns=target_frame.f_locals | target_frame.f_globals | {"__tb": lambda: print(tb_msg)}
    )

    return 1


if __name__ == '__main__':
    sys.exit(main())
