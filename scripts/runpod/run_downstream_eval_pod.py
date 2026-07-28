"""
End-to-end: spin up a RunPod H100 SXM pod, upload the eval script + the
already-trained Llama-2-7B pruner checkpoints, install deps, run
eval_downstream_llama2_7b.py, rsync results back down, terminate the pod.

*** UNTESTED against a live RunPod account. *** I do not have a RunPod API
key and cannot verify this end-to-end. Two things specifically are the most
likely to have drifted from what I remember of the `runpod` Python SDK and
should be checked against a live account / the SDK's current docs before
trusting this:
  1. GPU-type lookup (find_gpu_type_id below) -- RunPod's catalog ID strings
     for "H100 SXM" have changed naming before. This queries the live API
     and substring-matches rather than hardcoding an ID, specifically to
     survive that kind of drift, but the get_gpus() response schema itself
     could have changed field names.
  2. wait_for_running()'s parsing of pod["runtime"]["ports"] to extract the
     public IP + SSH port -- this is the field RunPod's API uses to report
     port-forwarded access, but schema details (key names, whether SSH is
     under a "tcp" or "public" section) are the part of their API most
     likely to differ from what's coded here.
Run --dry_run first (prints the resolved GPU type, image, and cost estimate,
creates nothing) before running for real.

COST SAFETY, read before running for real:
  - This creates a billed pod. It prints an estimated $/hr and requires
    typing "yes" to proceed (skip with --yes, only if you're scripting this
    unattended and have already verified the estimate).
  - Pod termination is in a `finally` block so it fires even if the eval
    run fails or times out (--eval_timeout, default 3h -- see the compute
    estimate in the conversation this was built from: ~1-2h expected, 3h
    leaves real margin without risking an unbounded runaway pod).
  - If this script itself is killed (Ctrl-C twice, SIGKILL, terminal
    closed) before the `finally` block runs, the pod is NOT automatically
    terminated. The pod ID and a manual termination command are printed
    immediately after creation, and again on every status poll, specifically
    so you have it on-screen if that happens. RunPod dashboard
    (https://www.runpod.io/console/pods) is the fallback if you lose the
    terminal entirely.
  - --keep_alive skips termination on purpose, for debugging a failed run.
    It prints a loud reminder every time it's used. Do not leave this on
    unattended.

WHAT GETS UPLOADED: scripts/hypernetwork/train/llm/{train_pruner_llama2_7b_wikitext2.py,
eval_downstream_llama2_7b.py} (eval script imports the training script, see
its own docstring) and the pruner.pt checkpoints from
experiments/latest/llama2_7b_wikitext2/llama2_7b_wikitext2_sweep/ (~1.6GB,
8 checkpoints x ~202MB -- this is the slow part of the upload, budget a
few minutes depending on your link).

USAGE:
  export RUNPOD_API_KEY=...
  export HF_TOKEN=...          # meta-llama/Llama-2-7b-hf is gate-licensed
  python3 scripts/runpod/run_downstream_eval_pod.py --dry_run
  python3 scripts/runpod/run_downstream_eval_pod.py
"""
import os
import sys
import time
import subprocess
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts", "hypernetwork", "train", "llm")
LOCAL_CHECKPOINT_DIR = os.path.join(
    REPO_ROOT, "experiments", "latest", "llama2_7b_wikitext2", "llama2_7b_wikitext2_sweep")
LOCAL_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements_llama2_downstream_runpod.txt")
LOCAL_OUT_DIR = os.path.join(REPO_ROOT, "experiments", "latest", "llama2_7b_downstream")

REMOTE_WORKSPACE = "/workspace"
REMOTE_SCRIPT_DIR = f"{REMOTE_WORKSPACE}/scripts"
REMOTE_SWEEP_DIR = f"{REMOTE_WORKSPACE}/results/llama2_7b_wikitext2_sweep"
REMOTE_OUT_DIR = f"{REMOTE_WORKSPACE}/results/llama2_7b_downstream"
REMOTE_REQUIREMENTS = f"{REMOTE_WORKSPACE}/requirements.txt"

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
ESTIMATED_USD_PER_HR = 3.5    # ballpark for H100 SXM secure-cloud on-demand as of when this was written -- CHECK against current RunPod pricing before trusting this number for a cost decision
ESTIMATED_HOURS = 1.5         # includes pod boot + upload + install + ~1-2h eval, see conversation


def find_gpu_type_id(query_substrings):
    import runpod
    gpus = runpod.get_gpus()
    for g in gpus:
        name = g.get("displayName", "") or g.get("id", "")
        if all(s.lower() in name.lower() for s in query_substrings):
            return g["id"], name
    available = "\n".join(f"  {g.get('id')!r}: {g.get('displayName')!r}" for g in gpus)
    raise RuntimeError(
        f"No GPU type matched {query_substrings!r}. Available:\n{available}\n"
        f"RunPod's catalog naming drifts -- pick the right id from this list "
        f"and pass it directly via --gpu_type_id to skip the search."
    )


def create_pod(args, gpu_type_id):
    import runpod
    env = {}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    pod = runpod.create_pod(
        name="keshava-llama2-7b-downstream-eval",
        image_name=args.image,
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        cloud_type="SECURE",
        container_disk_in_gb=20,
        volume_in_gb=args.volume_gb,
        volume_mount_path=REMOTE_WORKSPACE,
        ports="22/tcp",
        env=env,
    )
    return pod["id"]


def wait_for_running(pod_id, timeout_s=600, poll_every=10):
    import runpod
    print(f"Waiting for pod {pod_id} to report a reachable SSH port "
          f"(manual kill if needed: `python3 -c \"import runpod; runpod.api_key='...'; "
          f"runpod.terminate_pod('{pod_id}')\"`, or the RunPod dashboard) ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        info = runpod.get_pod(pod_id)
        runtime = info.get("runtime") or {}
        for p in runtime.get("ports", []):
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ip, port = p.get("ip"), p.get("publicPort")
                if ip and port:
                    print(f"  pod ready: ssh -p {port} root@{ip}", flush=True)
                    return ip, port
        print(f"  ... still waiting ({int(time.time()-t0)}s elapsed)", flush=True)
        time.sleep(poll_every)
    raise TimeoutError(f"Pod {pod_id} did not report a reachable SSH port within {timeout_s}s. "
                       f"Check the RunPod dashboard -- it may still be pulling the image.")


def ssh_base_args(ip, port, ssh_key):
    args = [
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
        # keepalives -- the eval command runs for 1-3h; without these, an idle
        # intermediate NAT/firewall can silently drop the connection long
        # before either side notices. ServerAliveCountMax=6 * Interval=30 ->
        # ssh gives up (and subprocess.run raises) within 3 minutes of the
        # link actually going dead, rather than hanging indefinitely.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
    ]
    if ssh_key:
        args += ["-i", ssh_key]
    return args


def rsync_up(local_path, remote_path, ip, port, ssh_key):
    ssh_cmd = "ssh " + " ".join(ssh_base_args(ip, port, ssh_key))
    cmd = ["rsync", "-avz", "--progress", "-e", ssh_cmd, local_path, f"root@{ip}:{remote_path}"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def rsync_down(remote_path, local_path, ip, port, ssh_key):
    ssh_cmd = "ssh " + " ".join(ssh_base_args(ip, port, ssh_key))
    os.makedirs(local_path, exist_ok=True)
    cmd = ["rsync", "-avz", "--progress", "-e", ssh_cmd, f"root@{ip}:{remote_path}/", local_path]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ssh_run(remote_cmd, ip, port, ssh_key, timeout_s=None):
    cmd = ["ssh"] + ssh_base_args(ip, port, ssh_key) + [f"root@{ip}", remote_cmd]
    print(f"  $ ssh root@{ip} '{remote_cmd}'", flush=True)
    subprocess.run(cmd, check=True, timeout=timeout_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_query", type=str, nargs="+", default=["H100", "SXM"],
                    help="Substrings to match against RunPod's GPU catalog displayName.")
    ap.add_argument("--gpu_type_id", type=str, default=None,
                    help="Skip the catalog search, use this exact RunPod GPU type id.")
    ap.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    ap.add_argument("--volume_gb", type=int, default=80,
                    help="Model weights (~14GB bf16) + HF cache + checkpoints (~2GB) + eval "
                         "datasets -- 80GB leaves real margin.")
    ap.add_argument("--ssh_key", type=str, default=os.path.expanduser("~/.ssh/id_ed25519"),
                    help="Must already be registered as a public key in your RunPod account "
                         "settings -- this script does not do that part.")
    ap.add_argument("--lambdas", type=str, default="0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0",
                    help="Passed through to eval_downstream_llama2_7b.py --lambdas.")
    ap.add_argument("--tasks", type=str,
                    default="piqa hellaswag winogrande arc_easy arc_challenge boolq openbookqa")
    ap.add_argument("--eval_timeout", type=int, default=3 * 3600,
                    help="Hard cap on the remote eval command, seconds. See module docstring "
                         "for why this exists -- pod termination still happens even if this fires.")
    ap.add_argument("--skip_sanity_check", action="store_true",
                    help="Skip the pre-flight sanity check and go straight to the full sweep. "
                         "Not recommended -- the sanity check is what catches an lm_eval API "
                         "mismatch in ~1 minute instead of ~90.")
    ap.add_argument("--boot_timeout", type=int, default=600)
    ap.add_argument("--keep_alive", action="store_true",
                    help="Skip termination at the end. For debugging only -- see module docstring.")
    ap.add_argument("--yes", action="store_true", help="Skip the cost-confirmation prompt.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Resolve GPU type + print the plan, create nothing.")
    args = ap.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        print("RUNPOD_API_KEY not set. export RUNPOD_API_KEY=... and retry.", file=sys.stderr, flush=True)
        sys.exit(1)
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set locally -- it won't be passed to the pod, and "
              "meta-llama/Llama-2-7b-hf is gate-licensed. The remote run will fail at model "
              "load without it.", flush=True)

    import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    if args.gpu_type_id:
        gpu_type_id, gpu_name = args.gpu_type_id, args.gpu_type_id
    else:
        gpu_type_id, gpu_name = find_gpu_type_id(args.gpu_query)

    print(f"Resolved GPU type: {gpu_type_id!r} ({gpu_name})", flush=True)
    print(f"Image: {args.image}", flush=True)
    print(f"Estimated cost: ~${ESTIMATED_USD_PER_HR}/hr x ~{ESTIMATED_HOURS}h "
          f"= ~${ESTIMATED_USD_PER_HR*ESTIMATED_HOURS:.2f} "
          f"(CHECK against RunPod's current on-demand H100 SXM price before trusting this)", flush=True)
    print(f"Uploading: {LOCAL_SCRIPT_DIR}, {LOCAL_CHECKPOINT_DIR} (~1.6GB)", flush=True)
    print(f"Downloading to: {LOCAL_OUT_DIR}", flush=True)

    if args.dry_run:
        print("\n--dry_run: stopping here, no pod created.", flush=True)
        return

    if not args.yes:
        resp = input(f"\nCreate a billed pod now? [yes/N]: ").strip().lower()
        if resp != "yes":
            print("Aborted, no pod created.", flush=True)
            return

    pod_id = create_pod(args, gpu_type_id)
    print(f"\nPod created: {pod_id}", flush=True)
    print(f"Manual kill command (save this): "
          f"python3 -c \"import runpod; runpod.api_key='<key>'; runpod.terminate_pod('{pod_id}')\"",
          flush=True)

    try:
        ip, port = wait_for_running(pod_id, timeout_s=args.boot_timeout)

        print("\nInstalling rsync on the pod (runpod/pytorch:* images don't ship it by default "
              "-- this is what a rsync exit code 127 over ssh means) ...", flush=True)
        ssh_run("DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
               "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync",
               ip, port, args.ssh_key)

        print("\nUploading code + checkpoints ...", flush=True)
        ssh_run(f"mkdir -p {REMOTE_SCRIPT_DIR} {REMOTE_SWEEP_DIR} {REMOTE_OUT_DIR}", ip, port, args.ssh_key)
        rsync_up(LOCAL_SCRIPT_DIR + "/", REMOTE_SCRIPT_DIR + "/", ip, port, args.ssh_key)
        rsync_up(LOCAL_CHECKPOINT_DIR + "/", REMOTE_SWEEP_DIR + "/", ip, port, args.ssh_key)
        rsync_up(LOCAL_REQUIREMENTS, REMOTE_REQUIREMENTS, ip, port, args.ssh_key)

        print("\nInstalling deps ...", flush=True)
        print("Remote output streams live below (PYTHONUNBUFFERED + python3 -u -- without "
              "these, a non-interactive SSH pipe block-buffers Python's stdout and you'd see "
              "nothing until the whole command finished):\n", flush=True)
        env_prefix = f"cd {REMOTE_SCRIPT_DIR} && export PYTHONUNBUFFERED=1 && HF_TOKEN={os.environ.get('HF_TOKEN', '')}"
        eval_args = (
            f"--sweep_dir {REMOTE_SWEEP_DIR} --lambdas {args.lambdas} "
            f"--tasks {args.tasks} --out_dir {REMOTE_OUT_DIR}"
        )
        ssh_run(f"{env_prefix} pip install -q -r {REMOTE_REQUIREMENTS}", ip, port, args.ssh_key)

        if not args.skip_sanity_check:
            print("\nSanity check (1 task, 5 examples, dense model only -- catches lm_eval "
                  "API mismatches in under a minute instead of 90 minutes into the real "
                  "run) ...", flush=True)
            ssh_run(f"{env_prefix} python3 -u eval_downstream_llama2_7b.py {eval_args} --sanity_check",
                   ip, port, args.ssh_key, timeout_s=300)
            print("  sanity check passed.", flush=True)

        print("\nRunning full eval (this is the slow part -- see the compute estimate in "
              "the conversation this was built from, roughly 1-2h) ...", flush=True)
        ssh_run(f"{env_prefix} python3 -u eval_downstream_llama2_7b.py {eval_args}",
               ip, port, args.ssh_key, timeout_s=args.eval_timeout)

        print("\nDownloading results ...", flush=True)
        rsync_down(REMOTE_OUT_DIR, LOCAL_OUT_DIR, ip, port, args.ssh_key)
        print(f"\nResults -> {LOCAL_OUT_DIR}/", flush=True)

    finally:
        if args.keep_alive:
            print(f"\n*** --keep_alive set: pod {pod_id} is STILL RUNNING AND BILLING. ***\n"
                  f"Terminate manually when done: "
                  f"python3 -c \"import runpod; runpod.api_key='<key>'; runpod.terminate_pod('{pod_id}')\"",
                  flush=True)
        else:
            print(f"\nTerminating pod {pod_id} ...", flush=True)
            try:
                runpod.terminate_pod(pod_id)
                print("  terminated.", flush=True)
            except Exception as e:
                print(f"  TERMINATION FAILED: {e}\n"
                      f"  Pod {pod_id} may still be running and billing -- check "
                      f"https://www.runpod.io/console/pods manually.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
