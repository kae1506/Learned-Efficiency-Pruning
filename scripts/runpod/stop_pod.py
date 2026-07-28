"""
Kill a RunPod pod from the inside, over SSH -- no RUNPOD_API_KEY needed.

Runs `kill -9 1` on the pod: kills PID 1 (the container's main process),
which makes the container exit. RunPod's supervisor detects the exited
container and stops it. This is the SSH-only kill, as opposed to
scripts/runpod/run_downstream_eval_pod.py's API-based
`runpod.terminate_pod(pod_id)` (needs RUNPOD_API_KEY, deletes the pod
outright via RunPod's API rather than crashing it from inside).

CAVEAT, genuinely uncertain, flagging rather than asserting: killing PID 1
stops the container (and with it, GPU compute -- the expensive part of the
bill), but I don't know for certain whether RunPod's billing treats a
self-exited container identically to an API-terminated one for every pod/
billing type (On-Demand vs Spot, Secure vs Community Cloud). Check the
RunPod dashboard (https://www.runpod.io/console/pods) after running this to
confirm the pod actually shows as stopped, not just "Exited" with a
reservation still held -- if it's still there, the API-based
runpod.terminate_pod() (or the dashboard's own Terminate button) is the
sure way to fully delete it.

USAGE:
  python3 scripts/runpod/stop_pod.py <ip> <port> [--ssh_key PATH]
  # e.g. from the "ssh -p 15623 root@103.207.149.105" you've been using:
  python3 scripts/runpod/stop_pod.py 103.207.149.105 15623
"""
import os
import sys
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ip")
    ap.add_argument("port", type=int)
    ap.add_argument("--ssh_key", type=str, default=os.path.expanduser("~/.ssh/id_ed25519"))
    args = ap.parse_args()

    ssh_cmd = [
        "ssh", "-p", str(args.port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
        "-i", args.ssh_key,
        f"root@{args.ip}",
        "kill -9 1",
    ]
    print(f"$ {' '.join(ssh_cmd)}", flush=True)
    try:
        # the ssh connection itself gets torn down as a side effect of PID 1
        # dying -- a non-zero/broken-pipe exit here is the EXPECTED outcome,
        # not a failure, so this doesn't check=True.
        subprocess.run(ssh_cmd, timeout=30)
        print("Sent. Container should be exiting now -- check "
              "https://www.runpod.io/console/pods to confirm it's actually "
              "stopped (see this script's docstring caveat on billing).", flush=True)
    except subprocess.TimeoutExpired:
        print("SSH command timed out -- pod may already be down, or unreachable. "
              "Check the RunPod dashboard.", flush=True)


if __name__ == "__main__":
    main()
