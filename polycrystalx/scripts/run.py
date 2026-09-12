"""Unified script to run a single job or a suite of jobs."""
import sys
import os
import argparse
import tempfile
import pathlib
import subprocess
import pickle

from . import get_input_module
from polycrystalx import inputs
from polycrystalx import processes


def is_mpi_active():
    """Check if running in an active MPI process."""
    mpi_env_vars = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
    )
    return any(var in os.environ for var in mpi_env_vars)


def resolve_attribute(args):
    """Resolve attribute name with proper defaults."""
    if args.attribute is not None:
        return args.attribute
    return "job_keys" if args.suite else "job"


def run_worker(input_module_name, attribute, key_file=None):
    """Execute a job within an MPI rank."""
    user_module = get_input_module(input_module_name)

    if key_file:
        with open(key_file, "rb") as f:
            key = pickle.load(f)
        if hasattr(user_module, "get_job"):
            job = user_module.get_job(key)
        else:
            job = key
    else:
        if not hasattr(user_module, attribute):
            raise AttributeError(f"module has no '{attribute}' attribute")
        job = getattr(user_module, attribute)
        if callable(job) and not isinstance(job, inputs.job.Job):
            job = job()
        elif hasattr(user_module, "get_job") and not isinstance(job, inputs.job.Job):
            if isinstance(job, (tuple, list, str, dict)):
                job = user_module.get_job(job)

    processes.run(job)


def run_single_job(args, attribute):
    """Manage execution of a single job with mpirun."""
    # Validate module and attribute before spawning MPI processes
    user_module = get_input_module(args.input_module)
    if not hasattr(user_module, attribute):
        raise AttributeError(f"module has no '{attribute}' attribute")

    cmd = [
        "mpirun", "-n", str(args.n),
        sys.executable, "-m", "polycrystalx.scripts.run",
        "--worker",
        args.input_module,
        attribute,
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def run_suite(args, attribute):
    """Manage execution of a suite of jobs with mpirun."""
    user_module = get_input_module(args.input_module)
    if not hasattr(user_module, attribute):
        emsg = f"job keys attribute '{attribute}' was not found"
        raise AttributeError(emsg)

    print("attribute: ", attribute)

    for job in getattr(user_module, attribute):
        # Pickle to a temp file.
        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with open(fp.name, "wb") as f:
                pickle.dump(job, f)

            print("\n===== New Job Starting", flush=True)

            # Now run MPI job.
            cmd = [
                "mpirun", "-n", str(args.n),
                sys.executable, "-m", "polycrystalx.scripts.run",
                "--worker",
                "--key-file", fp.name,
                args.input_module,
                attribute,
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"Job failed with exit code {res.returncode}", file=sys.stderr)
        finally:
            pathlib.Path(fp.name).unlink(missing_ok=True)


def main():
    """Main entry point for pxx_run."""
    p = argparser()
    args = p.parse_args()
    attribute = resolve_attribute(args)

    if args.worker or is_mpi_active():
        if args.suite and not args.worker:
            raise RuntimeError("Suites cannot be launched directly with mpirun. Run pxx_run without mpirun.")
        run_worker(args.input_module, attribute, key_file=args.key_file)
    elif args.suite:
        run_suite(args, attribute)
    else:
        run_single_job(args, attribute)


def argparser():
    """Construct command-line argument parser."""
    p = argparse.ArgumentParser(
        description="run a polycrystalx job or suite of jobs"
    )
    p.add_argument(
        'input_module', type=str,
        help="module with inputs (name or path)"
    )
    p.add_argument(
        'attribute', type=str, nargs='?', default=None,
        help="name of attribute to run; defaults to 'job' for a single job and to 'job_keys' for a suite of jobs"
    )
    p.add_argument(
        '-n', type=int, default=2, metavar='N',
        help="number of processes to use"
    )
    p.add_argument(
        '-s', '--suite', action='store_true',
        help="run a suite of jobs; if not specified then run a single job"
    )
    # Internal options for MPI worker execution
    p.add_argument(
        '--worker', action='store_true', help=argparse.SUPPRESS
    )
    p.add_argument(
        '--key-file', type=str, default=None, help=argparse.SUPPRESS
    )

    return p


if __name__ == "__main__":
    main()
