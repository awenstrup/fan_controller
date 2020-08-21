"""Run a script on the raspberry pi.

Requires an argument, "run" or "upload", specifiying
the script to be run on the pi

Assumes the machine calling this script is authorized
to ssh into the pi without a password.
"""
# Imports

# Base Python
import sys
import os
import os.path as path
import logging

# Paramiko didn't support the file transfer operations I needed, so just using os.system
# import paramiko

import paths

USERNAME = "pi"
HOSTNAME = "raspberrypi"
PI = f"{USERNAME}@{HOSTNAME}"

# Create Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Helpers
def run_remote(command: str):
    return os.system(f"ssh {PI} " + '"' + command + '"')


def run_local(command: str):
    return os.system(command)


def copy_files():
    commands = "rm -r fan_controller; "
    commands += "mkdir fan_controller; "
    commands += "touch fan_controller/__init__.py; "
    commands += "mkdir fan_controller/src; "
    commands += "touch fan_controller/src/__init__.py; "
    run_remote(commands)

    commands = f"scp -r {paths.PI} {PI}:~/fan_controller/src; "
    commands += f"scp {paths.SRC}/paths.py {PI}:~/fan_controller/src; "
    commands += f"scp -r {paths.LITE_MODEL} {PI}:~/fan_controller; "
    run_local(commands)


def install_requirements():
    logger.info("Installing requirements...")
    command = "pip3 install -r fan_controller/src/pi/requirements.txt;"
    run_remote(command)
    logger.info("Done installing requirements!")


def run_script():
    try:
        script = sys.argv[1] + ".py"
    except:
        raise Exception("No script specified. Call with either 'upload' or 'run'")

    if script not in ["upload.py", "run.py"]:
        raise Exception(f"Invalid script specified: {script}. Call with either 'upload' or 'run'")

    command = f"python3 fan_controller/src/pi/{script};"
    logger.info(f"Running: {command}")
    run_remote(command)


if __name__ == "__main__":
    copy_files()
    install_requirements()
    run_script()
