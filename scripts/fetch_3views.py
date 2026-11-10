"""Download and unpack the three-view renders produced by render_3views.sh.

Credentials come from the environment only (same contract as ssh_exec.py):
    SCNET_SSH_HOST, SCNET_SSH_PORT, SCNET_SSH_USER, SCNET_SSH_PASS

Usage:
    python fetch_3views.py

Writes D:\\3dgs\\proj-D6\\renders_3v\\{clean,q20,gt}\\{00000,00009,00018}.png
"""
import os
import sys
import tarfile

import paramiko

HOST = os.environ["SCNET_SSH_HOST"]
PORT = int(os.environ["SCNET_SSH_PORT"])
USER = os.environ.get("SCNET_SSH_USER", "root")
PWD = os.environ["SCNET_SSH_PASS"]

REMOTE = "/public/home/qq18676607705/fig12_3v.tar.gz"
LOCAL_DIR = r"D:\3dgs\proj-D6\renders_3v"
LOCAL_TAR = os.path.join(LOCAL_DIR, "fig12_3v.tar.gz")


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD,
              timeout=25, banner_timeout=25, auth_timeout=25,
              look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    print("downloading", REMOTE)
    sftp.get(REMOTE, LOCAL_TAR)
    sftp.close()
    c.close()

    with tarfile.open(LOCAL_TAR, "r:gz") as tf:
        tf.extractall(LOCAL_DIR)
    # tar was created with -C $BASE fig12_3v, so files land in renders_3v/fig12_3v
    inner = os.path.join(LOCAL_DIR, "fig12_3v")
    for kind in ("clean", "q20", "gt"):
        d = os.path.join(inner, kind)
        if os.path.isdir(d):
            print(kind, sorted(os.listdir(d)))
        else:
            print("MISSING", d)
    print("OK ->", inner)


if __name__ == "__main__":
    sys.exit(main())
