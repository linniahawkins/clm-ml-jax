"""
The ``clm-ml`` command.

    clm-ml run run.yaml [--set key=value ...]   run the model and write outputs
    clm-ml check run.yaml [--set ...]           validate all files and print a summary
    clm-ml convert NAMELIST -o DIR              write run.yaml + model.yaml from a namelist

``--set`` changes one setting without editing files: ``--set time.days=7`` changes
run.yaml; ``--set model.physics.stomatal_conductance=medlyn`` changes model.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clm_ml.io.yaml_schema import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clm-ml", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("run", "run the model"), ("check", "validate and summarize")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("run_file", type=Path, help="path to run.yaml")
        p.add_argument("--set", dest="overrides", action="append", default=[],
                       metavar="KEY=VALUE", help="override one setting (repeatable)")  # fmt: skip

    p = sub.add_parser("convert", help="convert a Fortran-style namelist to YAML")
    p.add_argument("namelist", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            from clm_ml.legacy.namelist import convert_namelist

            for path in convert_namelist(args.namelist, args.out_dir).values():
                print(f"wrote {path}")
            return 0

        from clm_ml.run import load_run

        run = load_run(args.run_file, args.overrides)
        print(run.summary())
        if args.command == "run":
            run.execute()
            print(f"Outputs written to {run.output_directory}")
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
