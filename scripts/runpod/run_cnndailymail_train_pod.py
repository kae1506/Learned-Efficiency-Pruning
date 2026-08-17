"""
End-to-end: spin up a RunPod H100 SXM pod, upload
train_pruner_llama2_7b_cnndailymail.py, install deps, run the pruner
training sweep, rsync results back down, terminate the pod. Structurally
identical to scripts/runpod/run_downstream_eval_pod.py -- same tmux-detached
job pattern, same cost-safety gating, same pod-termination discipline -- see
that script's docstring for the full reasoning behind each of those choices
(SSH-connection-drop resilience, --keep_alive, manual-kill printouts, etc.),
not re-derived here.

*** UNTESTED against a live RunPod account, AND against a real run of
train_pruner_llama2_7b_cnndailymail.py itself. *** Same two RunPod-API
uncertainties flagged in run_downstream_eval_pod.py's docstring (GPU-type
lookup, wait_for_running()'s ports-field parsing) apply here unchanged.
Run --dry_run first.

COST SAFETY -- read before running for real:
  - This is TRAINING, not the downstream eval script's few-minutes-per-
    checkpoint scoring pass -- it is a genuinely long, genuinely uncertain
    run. Nothing about its wall-clock cost has been measured: max_steps=12000
    per (λ, seed), default 9 lambdas x 1 seed = 9 runs, EACH of which also
    does an expensive ROUGE-via-generation eval (300 test examples x 2
    [orig, pruned], autoregressive decoding -- much slower per example than
    the CE-based sibling scripts' scoring-only evals). The training script's
    own docstring flags all of this as unvalidated. DO NOT trust
    ESTIMATED_HOURS below as anything more than a placeholder -- it has not
    been checked against a real run.
  - Strongly recommended before committing to the full 9-lambda sweep:
    pass --lambdas with a single value (e.g. --lambdas 0.1) for a first real
    pod run, or rely on --skip_sanity_check=False (the default) so the
    identity-gate sanity check at least catches a broken hook/dtype path in
    under a minute before any real compute is spent. This script has no
    equivalent of the eval script's cheap sanity check for the FULL training
    loop -- --sanity_check on the remote script only validates the gate
    no-op, not that 12000 steps will actually converge or that ROUGE
    generation works end-to-end. Consider a short smoke test first (small
    --max_steps override) if you want that confidence before the real run.
  - Same pod-termination discipline as run_downstream_eval_pod.py: the
    `finally` block terminates the pod even on failure/timeout; the tmux-
    detached job survives a dropped local SSH connection; --keep_alive skips
    termination for debugging and prints a loud running-cost reminder every
    time it's used.
  - --train_timeout (default 12h) is a hard cap on the remote command. Given
    the total uncertainty above, this is a wide-margin guess, not a
    validated bound -- a real run could finish well under it, or could hit
    it and be killed mid-sweep. Check progress via the streamed tmux output;
    raise --train_timeout for a deliberately long unattended run only after
    you've seen at least one (λ, seed) run complete and know its real
    per-run time.

WHAT GETS UPLOADED: scripts/hypernetwork/train/llm/ (all of it, same as
run_downstream_eval_pod.py -- train_pruner_llama2_7b_cnndailymail.py is
self-contained and doesn't import from any sibling script, but uploading
the whole directory matches the existing convention and costs nothing extra
worth avoiding). No checkpoints to upload -- this trains from scratch.

USAGE:
  export RUNPOD_API_KEY=...
  export HF_TOKEN=...          # meta-llama/Llama-2-7b-hf is gate-licensed
  python3 scripts/runpod/run_cnndailymail_train_pod.py --dry_run
  python3 scripts/runpod/run_cnndailymail_train_pod.py --lambdas 0.1   # first real run, one lambda
  python3 scripts/runpod/run_cnndailymail_train_pod.py                # full default sweep
"""
import os
import sys
import time
import subprocess
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts", "hypernetwork", "train", "llm")
LOCAL_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements_llama2_cnndailymail_runpod.txt")
LOCAL_OUT_DIR = os.path.join(REPO_ROOT, "experiments", "latest", "llama2_7b_cnndailymail")

REMOTE_WORKSPACE = "/workspace"
REMOTE_SCRIPT_DIR = f"{REMOTE_WORKSPACE}/scripts"
REMOTE_OUT_DIR = f"{REMOTE_WORKSPACE}/results/llama2_7b_cnndailymail_sweep"
REMOTE_REQUIREMENTS = f"{REMOTE_WORKSPACE}/requirements.txt"
REMOTE_TRAIN_SCRIPT = f"{REMOTE_WORKSPACE}/run_train.sh"
REMOTE_TRAIN_LOG = f"{REMOTE_WORKSPACE}/train_live.log"
REMOTE_TRAIN_EXIT = f"{REMOTE_WORKSPACE}/train_exit_code"
TMUX_SESSION = "train_run"

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
ESTIMATED_USD_PER_HR = 3.5    # ballpark for H100 SXM secure-cloud on-demand as of when this was written -- CHECK against current RunPod pricing
ESTIMATED_HOURS = None        # deliberately unset -- see COST SAFETY above, no real run to base this on


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
        name="keshava-llama2-7b-cnndailymail-train",
        image_name=args.image,
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        cloud_type="SECURE",
        # No volume_in_gb/volume_mount_path -- see run_downstream_eval_pod.py's
        # create_pod() comment for why (network-volume chown restriction).
        container_disk_in_gb=args.container_disk_gb,
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
        # keepalives -- the training run can go for hours; without these, an
        # idle intermediate NAT/firewall can silently drop the connection
        # long before either side notices.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
    ]
    if ssh_key:
        args += ["-i", ssh_key]
    return args


def rsync_up(local_path, remote_path, ip, port, ssh_key):
    ssh_cmd = "ssh " + " ".join(ssh_base_args(ip, port, ssh_key))
    # --no-owner --no-group -- see run_downstream_eval_pod.py's rsync_up() comment.
    cmd = ["rsync", "-avz", "--no-owner", "--no-group", "--progress", "-e", ssh_cmd,
          local_path, f"root@{ip}:{remote_path}"]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def rsync_down(remote_path, local_path, ip, port, ssh_key):
    ssh_cmd = "ssh " + " ".join(ssh_base_args(ip, port, ssh_key))
    os.makedirs(local_path, exist_ok=True)
    cmd = ["rsync", "-avz", "--no-owner", "--no-group", "--progress", "-e", ssh_cmd,
          f"root@{ip}:{remote_path}/", local_path]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ssh_run(remote_cmd, ip, port, ssh_key, timeout_s=None):
    cmd = ["ssh"] + ssh_base_args(ip, port, ssh_key) + [f"root@{ip}", remote_cmd]
    print(f"  $ ssh root@{ip} '{remote_cmd}'", flush=True)
    subprocess.run(cmd, check=True, timeout=timeout_s)


def ssh_write_file(remote_path, content, ip, port, ssh_key, timeout_s=30):
    cmd = ["ssh"] + ssh_base_args(ip, port, ssh_key) + [f"root@{ip}", f"cat > {remote_path}"]
    subprocess.run(cmd, input=content.encode(), check=True, timeout=timeout_s)


def ssh_capture(remote_cmd, ip, port, ssh_key, timeout_s=30):
    cmd = ["ssh"] + ssh_base_args(ip, port, ssh_key) + [f"root@{ip}", remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def start_tmux_job(remote_cmd, ip, port, ssh_key):
    """See run_downstream_eval_pod.py's start_tmux_job() docstring -- same
    detached-tmux pattern, same reasoning (a dropped local SSH connection
    must not kill an in-progress multi-hour run).

    Output is piped through `tee` rather than redirected straight to a file,
    so it's visible BOTH in a live `tmux attach` (for interactive debugging
    while the run is in progress) AND in REMOTE_TRAIN_LOG (for
    stream_tmux_job's polling, and for whatever's on disk if the poll loop
    itself dies). `set -o pipefail` is required for this to work correctly:
    without it, `$?` after a pipeline reflects tee's exit status (always 0),
    not the training command's -- REMOTE_TRAIN_EXIT would then always read
    0 even on a real failure."""
    script = (
        "#!/bin/bash\n"
        "set -o pipefail\n"
        f"{remote_cmd} 2>&1 | tee {REMOTE_TRAIN_LOG}\n"
        f"echo $? > {REMOTE_TRAIN_EXIT}\n"
    )
    ssh_write_file(REMOTE_TRAIN_SCRIPT, script, ip, port, ssh_key)
    ssh_run(f"chmod +x {REMOTE_TRAIN_SCRIPT} && rm -f {REMOTE_TRAIN_EXIT} {REMOTE_TRAIN_LOG} && "
            f"tmux new-session -d -s {TMUX_SESSION} {REMOTE_TRAIN_SCRIPT}",
            ip, port, ssh_key)
    print(f"  started in tmux session {TMUX_SESSION!r} -- output is tee'd to both the tmux "
          f"pane and {REMOTE_TRAIN_LOG}, so attaching live shows exactly what this script is "
          f"polling. If this script's own connection drops, the job keeps running -- "
          f"reattach manually with:\n"
          f"    ssh -p {port} root@{ip} -t 'tmux attach -t {TMUX_SESSION}'", flush=True)


def close_tmux_session(ip, port, ssh_key):
    """Explicit teardown step, run before pod termination. Safe/idempotent
    even if the session has already exited on its own (tmux closes a
    session once its command finishes, by default) -- `; true` forces the
    remote command to always report success so this never raises on a
    missing session."""
    try:
        ssh_run(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; true", ip, port, ssh_key, timeout_s=30)
    except Exception as e:
        print(f"  (non-fatal) could not close tmux session {TMUX_SESSION!r}: {e}", flush=True)


def stream_tmux_job(ip, port, ssh_key, timeout_s, poll_every=15):
    """See run_downstream_eval_pod.py's stream_tmux_job() docstring -- same
    poll-don't-hold-a-connection-open pattern."""
    seen_bytes = 0
    t0 = time.time()
    consecutive_failures = 0
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"Remote tmux job did not finish within {timeout_s}s. It may still be "
                f"running -- check with: ssh -p {port} root@{ip} -t 'tmux attach -t {TMUX_SESSION}'")
        try:
            r = ssh_capture(
                f"tail -c +{seen_bytes + 1} {REMOTE_TRAIN_LOG} 2>/dev/null; "
                f"echo __POLL_SPLIT__; cat {REMOTE_TRAIN_EXIT} 2>/dev/null || true",
                ip, port, ssh_key, timeout_s=30)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"  [poll #{consecutive_failures} failed ({type(e).__name__}: {e}) -- "
                  f"job keeps running in tmux regardless of this, retrying in {poll_every}s]",
                  flush=True)
            time.sleep(poll_every)
            continue

        new_text, _, exit_text = r.stdout.partition("__POLL_SPLIT__\n")
        if new_text:
            sys.stdout.write(new_text)
            sys.stdout.flush()
            seen_bytes += len(new_text.encode())

        exit_text = exit_text.strip()
        if exit_text:
            return int(exit_text)
        time.sleep(poll_every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_query", type=str, nargs="+", default=["H100", "SXM"],
                    help="Substrings to match against RunPod's GPU catalog displayName.")
    ap.add_argument("--gpu_type_id", type=str, default=None,
                    help="Skip the catalog search, use this exact RunPod GPU type id.")
    ap.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    ap.add_argument("--container_disk_gb", type=int, default=100,
                    help="Ordinary container disk backing /workspace. Model weights "
                         "(~14GB bf16) + HF cache + CNN/DailyMail dataset + checkpoints "
                         "(up to 9 lambdas x ~200MB) -- 100GB leaves real margin, a bit "
                         "more than the downstream-eval script's 80GB given the extra "
                         "dataset download.")
    ap.add_argument("--ssh_key", type=str, default=os.path.expanduser("~/.ssh/id_ed25519"),
                    help="Must already be registered as a public key in your RunPod account "
                         "settings -- this script does not do that part.")
    ap.add_argument("--lambdas", type=str, default="0.01 0.05 0.1 0.2 0.25 0.3 0.4 0.8 1.6",
                    help="Passed through to train_pruner_llama2_7b_cnndailymail.py --lambdas. "
                         "Strongly consider a single value (e.g. '0.1') for a first real run -- "
                         "see module docstring's COST SAFETY section.")
    ap.add_argument("--seeds", type=str, default="0",
                    help="Passed through as --seeds.")
    ap.add_argument("--max_steps", type=int, default=12000,
                    help="Passed through as --max_steps. Matches the training script's own "
                         "current default (raised from 8000 per explicit instruction).")
    ap.add_argument("--extra_train_args", type=str, default="",
                    help="Any additional flags appended verbatim to the remote train command "
                         "(e.g. '--rouge_eval_examples 100' for a cheaper first run).")
    ap.add_argument("--train_timeout", type=int, default=12 * 3600,
                    help="Hard cap on the remote training command, seconds. Wide-margin "
                         "placeholder -- see module docstring's COST SAFETY section, this "
                         "has not been validated against a real run.")
    ap.add_argument("--skip_sanity_check", action="store_true",
                    help="Skip the pre-flight identity-gate sanity check and go straight to "
                         "the full sweep. Not recommended -- it catches a broken hook/dtype "
                         "path in under a minute instead of hours into the real run.")
    ap.add_argument("--boot_timeout", type=int, default=600)
    ap.add_argument("--keep_alive", action="store_true",
                    help="Skip termination at the end. For debugging only.")
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
    if ESTIMATED_HOURS is None:
        print(f"Estimated cost: ~${ESTIMATED_USD_PER_HR}/hr x UNKNOWN duration -- "
              f"no validated per-run time exists yet for this script (see module docstring's "
              f"COST SAFETY section). Do not commit to the full {args.lambdas!r} sweep without "
              f"first watching at least one (λ, seed) run complete.", flush=True)
    print(f"Uploading: {LOCAL_SCRIPT_DIR}", flush=True)
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

        print("\nInstalling rsync + tmux on the pod (runpod/pytorch:* images don't ship "
              "either by default) ...", flush=True)
        ssh_run("DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
               "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux",
               ip, port, args.ssh_key)

        print("\nUploading code ...", flush=True)
        ssh_run(f"mkdir -p {REMOTE_SCRIPT_DIR} {REMOTE_OUT_DIR}", ip, port, args.ssh_key)
        rsync_up(LOCAL_SCRIPT_DIR + "/", REMOTE_SCRIPT_DIR + "/", ip, port, args.ssh_key)
        rsync_up(LOCAL_REQUIREMENTS, REMOTE_REQUIREMENTS, ip, port, args.ssh_key)

        print("\nInstalling deps ...", flush=True)
        print("Remote output streams live below (PYTHONUNBUFFERED + python3 -u) ...\n", flush=True)
        env_prefix = f"cd {REMOTE_SCRIPT_DIR} && export PYTHONUNBUFFERED=1 && HF_TOKEN={os.environ.get('HF_TOKEN', '')}"
        train_args = (
            f"--lambdas {args.lambdas} --seeds {args.seeds} --max_steps {args.max_steps} "
            f"--out_dir {REMOTE_OUT_DIR} {args.extra_train_args}"
        ).strip()
        ssh_run(f"{env_prefix} pip install -q -r {REMOTE_REQUIREMENTS}", ip, port, args.ssh_key)

        if not args.skip_sanity_check:
            print("\nSanity check (identity-gate no-op check on a validation slice -- catches "
                  "a broken hook/dtype path in under a minute) ...", flush=True)
            ssh_run(f"{env_prefix} python3 -u train_pruner_llama2_7b_cnndailymail.py "
                   f"--out_dir {REMOTE_OUT_DIR} --sanity_check",
                   ip, port, args.ssh_key, timeout_s=300)
            print("  sanity check passed.", flush=True)

        print(f"\nRunning training sweep in a detached tmux session -- duration UNKNOWN, "
              f"see module docstring's COST SAFETY section. A dropped local connection does "
              f"not kill this run (see start_tmux_job) ...", flush=True)
        start_tmux_job(f"{env_prefix} python3 -u train_pruner_llama2_7b_cnndailymail.py {train_args}",
                       ip, port, args.ssh_key)
        exit_code = stream_tmux_job(ip, port, args.ssh_key, timeout_s=args.train_timeout)
        if exit_code != 0:
            raise RuntimeError(
                f"Remote training (tmux session {TMUX_SESSION!r}) exited with code {exit_code}. "
                f"See the streamed output above for the failure -- full log is still on the "
                f"pod at {REMOTE_TRAIN_LOG} until termination below.")

        print("\nDownloading results ...", flush=True)
        rsync_down(REMOTE_OUT_DIR, LOCAL_OUT_DIR, ip, port, args.ssh_key)
        print(f"\nResults -> {LOCAL_OUT_DIR}/", flush=True)

    finally:
        if "ip" in locals() and "port" in locals():
            print(f"\nClosing tmux session {TMUX_SESSION!r} ...", flush=True)
            close_tmux_session(ip, port, args.ssh_key)
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
