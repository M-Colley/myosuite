import os
import shutil
from os.path import expanduser

import git

curr_dir = os.path.dirname(os.path.abspath(__file__))
simhive_path = os.path.join(curr_dir, 'myosuite', 'simhive')


# from myosuite.utils.import_utils import fetch_git

def fetch_git(repo_url, commit_hash, clone_directory, clone_path=None):
    """
    fetch git repo using provided details
    """
    if clone_path is None:
        clone_path = os.path.join(expanduser("~"), ".myosuite")
    clone_directory = os.path.join(clone_path, clone_directory)

    os.makedirs(clone_directory, exist_ok=True)

    if not os.path.exists(os.path.join(clone_directory, '.git')):
        repo = git.Repo.clone_from(repo_url, clone_directory)
        print(f"{repo_url} cloned at {clone_directory}")
    else:
        repo = git.Repo(clone_directory)
        repo.remote('origin').fetch()

    if repo.head.commit.hexsha != commit_hash:
        repo.git.checkout(commit_hash)
        print(f"{repo_url}@{commit_hash} fetched at {clone_directory}")

    if repo.head.commit.hexsha != commit_hash:
        raise RuntimeError(f"Failed to check out pinned commit {commit_hash}.")

    return clone_directory


def clean_simhive():
    """
    Remove cached simhive if it exists
    """
    print("MyoSuite:> Clearing SimHive ...")
    api_path = os.path.join(simhive_path, 'myo_model')
    if os.path.exists(api_path):
        shutil.rmtree(api_path)
    else:
        print("MyoSuite:> SimHive/myo_model directory does not exist.")
    print("MyoSuite:> SimHive cleared")


def accept_license():
    prompt = """
A permissive license for non-commercial scientific research is available.
You can review the license at: https://github.com/myolab/myo_model/blob/main/LICENSE

Do you accept the terms of the license? (yes/no):
"""
    while True:
        response = input(prompt).strip().lower()
        if response == 'yes':
            print("Thank you for accepting the license. You may proceed.")
            return True
        if response == 'no':
            print("You have rejected the license terms. Exiting...")
            return False
        print("Invalid input. Please enter 'yes' or 'no'.")


def fetch_simhive():
    """
    fetch a copy of simhive
    """
    print("MyoSuite:> Initializing...")

    # Mark the SimHive version (ToDo: Remove this when commits hashes are auto fetched from submodules)
    __version__ = "2.5.0"

    # Inform user about API
    if accept_license():
        # Proceed with the rest of the code
        print("MyoSuite:> License accepted. Proceeding initialization ...")
    else:
        # Exit or handle the rejection case
        print("MyoSuite:> License rejected. Exiting")
        return

    # Fetch SimHive
    print("MyoSuite:> Downloading simulation assets (upto ~100MBs)")
    asset_path = fetch_git(
        repo_url="https://github.com/myolab/myo_model.git",
        commit_hash="619b1a876113e91a302b9baeaad6c2341e12ac81",
        clone_directory="myo_model",
        clone_path=simhive_path,
    )
    required_asset = os.path.join(asset_path, 'myoskeleton', 'myoskeleton.xml')
    if not os.path.isfile(required_asset):
        raise FileNotFoundError(f"Required simulation asset not found: {required_asset}")


    # mark successful creation of simhive
    filename = os.path.join(simhive_path, "simhive-version")
    with open(filename, 'w') as file:
        file.write(__version__)

    print("MyoSuite:> Successfully Initialized.")


if __name__ == "__main__":
    fetch_simhive()