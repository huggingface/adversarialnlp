from typing import Dict
import argparse
import logging

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules

from adversarialnlp import __version__
from adversarialnlp.commands.test_install import TestInstall

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main(prog: str = None,
         subcommand_overrides: Dict[str, Subcommand] = {}) -> None:
    """
    :mod:`~adversarialnlp.run` command.
    """
    # pylint: disable=dangerous-default-value
    parser = argparse.ArgumentParser(description="Run AdversarialNLP", usage='%(prog)s', prog=prog)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "test-install": TestInstall(),

            # Superseded by overrides
            **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
