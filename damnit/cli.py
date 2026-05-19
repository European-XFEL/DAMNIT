import argparse

import inspect
import logging
import os
import re
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


def get_existing_damnit_dir(proposal_or_dir: str | None):
    if proposal_or_dir is None:
        return None
    elif (path := Path(proposal_or_dir)).is_dir():
        return path
    elif proposal_or_dir.isdigit():
        proposal_name = f"p{int(proposal_or_dir):06d}"
        return Path(find_proposal(proposal_name)) / "usr/Shared/amore"
    else:
        sys.exit(f"{proposal_or_dir} is not a proposal number or DAMNIT database directory")


def handle_config_args(args, kv, converters={}):
    key = args.key
    if key:
        key = key.replace('-', '_')
    if args.value and key in converters:
        args.value = converters[key](args.value)

    if args.delete:
        if not key:
            sys.exit("Error: no key specified to delete")
        del kv[key]
    elif key and (args.value is not None):
        if args.num:
            try:
                value = int(args.value)
            except ValueError:
                value = float(args.value)
        else:
            value = args.value
        kv[key] = value
    elif key:
        try:
            print(repr(kv[key]))
        except KeyError:
            sys.exit(f"Error: key {key} not found")
    else:
        for k, v in kv.items():
            print(f"{k}={v!r}")


class Subcommand:
    name: str
    help: str | None = None

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        pass

    @staticmethod
    def run(args: argparse.Namespace):
        raise NotImplementedError


class SubcommandGroup(Subcommand):
    subcmds: list
    cmd_dest: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subcmd_map = {sc.name: sc for sc in cls.subcmds}

    @classmethod
    def arguments(cls, parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(dest=cls.cmd_dest, required=True)
        for subcmd in cls.subcmds:
            sub_ap = subparsers.add_parser(subcmd.name, help=subcmd.help)
            subcmd.arguments(sub_ap)

    @classmethod
    def run(cls, args: argparse.Namespace):
        return cls.subcmd_map[getattr(args, cls.cmd_dest)].run(args)


class GuiSubcmd(Subcommand):
    name = "gui"
    help = "Launch application"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            'proposal_or_dir', nargs='?',
            help="Either a proposal number or a database directory."
        )
        parser.add_argument(
            '--no-kafka', action='store_true',
            help="Don't try connecting to XFEL's Kafka broker"
        )
        parser.add_argument(
            "--software-opengl", action="store_true",
            help="Force software OpenGL. Use this if displaying interactive Plotly plots shows a black screen."
                 "Active by default on Maxwell if not started on a display node."
        )

    @staticmethod
    def run(args: argparse.Namespace):
        context_dir = get_existing_damnit_dir(args.proposal_or_dir)

        from .gui.main_window import run_app
        return run_app(context_dir,
                       software_opengl=args.software_opengl,
                       connect_to_kafka=not args.no_kafka)


class ListenSubcmd(Subcommand):
    name = "listen"
    help = "Watch for new runs & extract data from them"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        listen_args_grp = parser.add_mutually_exclusive_group()
        listen_args_grp.add_argument(
            "listener_dir", type=Path, default=Path.cwd(), nargs="?",
            help="Path to the listener database directory"
        )
        listen_args_grp.add_argument(
            '--test', action='store_true',
            help="Manually enter 'migrated' runs for testing"
        )
        listen_args_grp.add_argument(
            '--daemonize', action='store_true',
            help="Start the listener under a separate process managed by supervisord."
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend import start_listener

        if args.daemonize:
            return start_listener(args.listener_dir)
        else:
            if args.test:
                from .backend.test_listener import listen
            else:
                from .backend.listener import listen

            return listen(args.listener_dir)


class ListenerConfigSubcmd(Subcommand):
    name = "config"
    help = "See or change the config for the DAMNIT listener"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            '-d', '--delete', action='store_true',
            help="Delete the specified key",
        )
        parser.add_argument(
            '--num', action='store_true',
            help="Set the given value as a number instead of a string"
        )
        parser.add_argument(
            'key', nargs='?',
            help="The config key to see/change. If not given, list the whole configuration"
        )
        parser.add_argument(
            'value', nargs='?',
            help="A new value for the given key"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.listener import ListenerDB

        db = ListenerDB(Path.cwd())
        handle_config_args(args, db.settings,
                           # Convert `static_mode` to a bool
                           dict(static_mode=lambda x: bool(int(x))))


class ListenerAddSubcmd(Subcommand):
    name = "add"
    help = "Add a database to monitor"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "proposal", type=int,
            help="Proposal number"
        )
        parser.add_argument(
            "db_dir", type=Path, nargs='?',
            help="Path to the database directory"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.listener import ListenerDB

        db = ListenerDB(Path.cwd())
        official_dir = Path(find_proposal(f"p{args.proposal:06d}")) / "usr/Shared/amore"

        if args.db_dir is None:
            db_dir = official_dir
            official = True
        else:
            db_dir = args.db_dir
            official = db_dir == official_dir

        db.add_proposal_db(args.proposal, db_dir, official=official)
        print(f"Added proposal {args.proposal} at {db_dir}")


class ListenerRmSubcmd(Subcommand):
    name = "rm"
    help = "Remove a database from being monitored"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "db_dir", type=Path,
            help="Path to the database directory to remove"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.listener import ListenerDB

        db = ListenerDB(Path.cwd())
        db.remove_proposal_db(args.db_dir)
        print(f"Removed database at {args.db_dir}")


class ListenerDatabasesSubcmd(Subcommand):
    name = "databases"
    help = "Display the DAMNIT databases currently being monitored"

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.listener import ListenerDB

        db = ListenerDB(Path.cwd())
        all_proposals = db.all_proposals()
        sorted_proposals = sorted(all_proposals.keys())
        for p in sorted_proposals:
            if len(all_proposals[p]) > 1:
                print(f"p{p}:")
                for x in all_proposals[p]:
                    print("   ", x.db_dir, "" if x.official else " (unofficial)", sep="")
            else:
                x = all_proposals[p][0]
                print(f"p{p}: {x.db_dir}", "" if x.official else "(unofficial)")


class ListenerSubcmd(SubcommandGroup):
    name = "listener"
    help = "Manage the DAMNIT listener."
    cmd_dest = "listener_subcmd"
    subcmds = [
        ListenerConfigSubcmd,
        ListenerAddSubcmd,
        ListenerRmSubcmd,
        ListenerDatabasesSubcmd,
    ]


class CombinerRunSubcmd(Subcommand):
    name = "run"
    help = "Run the DAMNIT combiner"

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.combine import main
        return main()


class CombinerNowSubcmd(Subcommand):
    name = "now"
    help=("Combine files in the specified DAMNIT directory now. "
          "Can cause corruption if the combiner is running at the same time.")

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument("db_dir", type=Path)

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.combine import gather_all_fragments
        gather_all_fragments(args.db_dir)


class CombinerSubcmd(SubcommandGroup):
    name = "combiner"
    help = "Manage the DAMNIT combiner."
    cmd_dest = "combiner_subcmd"
    subcmds = [
        CombinerRunSubcmd,
        CombinerNowSubcmd,
    ]


class ReprocessSubcmd(Subcommand):
    name = "reprocess"
    help = "Extract data from specified runs."

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--in', dest="proposal_or_dir",
            help="Proposal number or DAMNIT directory in which to run. By "
                 "default, uses the CWD."
        )
        parser.add_argument(
            '--proposal', type=int,
            help="Proposal number, e.g. 1234, to include data from another proposal. "
                 "By default, uses the proposal number configured in the database."
        )
        parser.add_argument(
            '--match', type=str, action="append", default=[],
            help="String to match against variable titles (case-insensitive). Not a regex, simply `str in var.title`."
        )
        parser.add_argument(
            '--param', type=str, action="append", default=[],
            help="Parameter to set for reprocessing, e.g. threshold=2.2 or normalise=true"
        )
        parser.add_argument(
            '--watch', action='store_true',
            help="Run jobs one-by-one with live output in the terminal"
        )
        parser.add_argument(
            '--direct', action='store_true',
            help="Run processing in subprocesses on this node, instead of via Slurm"
        )
        parser.add_argument(
            '--concurrent-jobs', type=int, default=-1,
            help="The maximum number of jobs that will run at once (default is the `concurrent_jobs` database setting)"
        )
        parser.add_argument(
            "--mock", action="store_true",
            help="Use a fake run object instead of loading one from disk."
                 " Note: do not use the passed `run` object in your context file with this"
                 " flag enabled, it will not contain any useful data."
        )
        parser.add_argument(
            'run', nargs='+',
            help="Run number, e.g. 96. Multiple runs can be specified at once, "
                 "or pass 'all' to reprocess all runs in the database."
        )

    @staticmethod
    def run(args: argparse.Namespace):
        # Hide some logging from Kafka to make things more readable
        logging.getLogger('kafka').setLevel(logging.WARNING)

        if damnit_dir := get_existing_damnit_dir(args.proposal_or_dir):
            os.chdir(damnit_dir)

        from .backend.extraction_control import reprocess
        reprocess(
            runs=args.run,
            proposal=args.proposal,
            match=args.match,
            params=args.param,
            mock=args.mock,
            watch=args.watch,
            direct=args.direct,
            limit_running=args.concurrent_jobs,
        )


class ReadctxSubcmd(Subcommand):
    name = "read-context"
    help="Re-read the context file and update variables in the database"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--no-kafka', action='store_true',
            help="Don't try connecting to XFEL's Kafka broker"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.extract_data import Extractor
        Extractor(connect_to_kafka=not args.no_kafka).update_db_vars()


class ProposalSubcmd(Subcommand):
    name = "proposal"
    help="Get or set the proposal number to collect metadata from"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            'proposal', nargs='?', type=int,
            help="Proposal number to set, e.g. 1234"
        )

    @staticmethod
    def run(args: argparse.Namespace):
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


class NewIdSubcmd(Subcommand):
    name = "new-id"
    help=("Set a new (random) database ID. Useful if a copy has been made "
          "which should not share an ID with the original.")

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            'db_dir', type=Path, default=Path.cwd(), nargs='?',
            help="Path to the database directory"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from secrets import token_hex
        from .backend.db import DamnitDB

        db = DamnitDB.from_dir(args.db_dir)
        db.metameta["db_id"] = token_hex(20)


class DbConfigSubcmd(Subcommand):
    name = "db-config"
    help = "See or change config in this database"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            '-d', '--delete', action='store_true',
            help="Delete the specified key",
        )
        parser.add_argument(
            '--num', action='store_true',
            help="Set the given value as a number instead of a string"
        )
        parser.add_argument(
            'key', nargs='?',
            help="The config key to see/change. If not given, list all config"
        )
        parser.add_argument(
            'value', nargs='?',
            help="A new value for the given key"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.db import DamnitDB

        db = DamnitDB()
        handle_config_args(args, db.metameta)


class MigrateSubcmd(Subcommand):
    name = "migrate"
    help = ("Execute migrations to help upgrading. Do NOT execute a migration "
            "unless you know what you're doing.")

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dry-run", action="store_true"
        )
        migrate_subparsers = parser.add_subparsers(dest="migrate_subcmd")
        migrate_subparsers.add_parser(
            "v0-to-v1",
            help="Migrate the SQLite database and HDF5 files from v0 to v1."
        )
        migrate_subparsers.add_parser(
            "intermediate-v1",
            help="Migrate the SQLite database HDF5 files from an initial implementation of v1 to the final"
                 " v1. Don't use this unless you know what you're doing."
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.db import DamnitDB
        from .migrations import migrate_intermediate_v1, migrate_v0_to_v1

        db = DamnitDB(allow_old=True)

        if args.migrate_subcmd == "v0-to-v1":
            migrate_v0_to_v1(db, Path.cwd(), args.dry_run)
        elif args.migrate_subcmd == "intermediate-v1":
            migrate_intermediate_v1(db, Path.cwd(), args.dry_run)


class InitSubcmd(Subcommand):
    name = "init"
    help = "Set up a new DAMNIT folder"

    @staticmethod
    def arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            'proposal_or_dir',
            help="Either a proposal number or a directory path."
        )
        parser.add_argument(
            '--proposal', type=int,
            help="Proposal number (if not in a proposal directory)"
        )
        parser.add_argument(
            '--like',
            help="Proposal number or DAMNIT directory to copy context file & "
                 "user-editable variables from"
        )
        parser.add_argument(
            '--context', type=Path,
            help="Context file to copy to the new location"
        )

    @staticmethod
    def run(args: argparse.Namespace):
        from .backend.db import initialize_proposal

        if args.proposal_or_dir.isdigit():
            proposal_num = int(args.proposal_or_dir)
            damnit_dir = Path(find_proposal(f"p{proposal_num:06d}")) / "usr/Shared/amore"
        else:
            damnit_dir = Path(args.proposal_or_dir).resolve()
            if args.proposal is not None:
                proposal_num = args.proposal
            elif m := re.match(r"/gpfs/exfel/u/.*/p([0-9]+)/", str(damnit_dir)):
                proposal_num = int(m[1])
            else:
                sys.exit("--proposal is required if not in a proposal directory")

        def check_file(p: Path):
            if not p.is_file():
                sys.exit(f"{p} is not a file")
            return p

        context_file_src = user_vars_src = None
        if like_dir := get_existing_damnit_dir(args.like):
            context_file_src = check_file(like_dir / "context.py")
            user_vars_src = check_file(like_dir / "runs.sqlite")

        if args.context is not None:
            context_file_src = check_file(args.context)

        damnit_dir.mkdir(parents=True, exist_ok=True)
        print(f"Setting up DAMNIT in {damnit_dir}")

        initialize_proposal(
            damnit_dir,
            proposal=proposal_num,
            context_file_src=context_file_src,
            user_vars_src=user_vars_src,
        )


def main(argv=None):
    # Check if script was called as amore-proto and show deprecation warning
    if Path(sys.argv[0]).name == 'amore-proto':
        print(colored("Warning: 'amore-proto' has been renamed to 'damnit'. Please update your scripts.", 'yellow'), file=sys.stderr)

    ap = ArgumentParser()
    ap.add_argument('--debug', action='store_true',
                    help="Show debug logs.")
    ap.add_argument('--debug-repl', action='store_true',
                    help="Drop into an IPython repl if an exception occurs. Local variables at the point of the exception will be available.")
    subparsers = ap.add_subparsers(required=True, dest='subcmd')
    subcmds = [
        GuiSubcmd,
        ListenSubcmd,
        ListenerSubcmd,
        CombinerSubcmd,
        ReprocessSubcmd,
        ReadctxSubcmd,
        ProposalSubcmd,
        NewIdSubcmd,
        DbConfigSubcmd,
        MigrateSubcmd,
        InitSubcmd,
    ]
    subcmd_map = {}
    for subcmd in subcmds:
        sub_ap = subparsers.add_parser(subcmd.name, help=subcmd.help)
        subcmd.arguments(sub_ap)
        subcmd_map[subcmd.name] = subcmd

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)-38s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.debug_repl:
        sys.excepthook = excepthook

    return subcmd_map[args.subcmd].run(args)


if __name__ == '__main__':
    sys.exit(main())
