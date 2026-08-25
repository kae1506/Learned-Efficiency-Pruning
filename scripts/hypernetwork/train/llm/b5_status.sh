#!/bin/bash
# One-line status for the B5 pruned-vs-dense LoRA run. Prints current arm, phase,
# step progress, throughput and ETA by parsing the launcher's log. Used as the
# command for a periodic Monitor; also fine to run by hand.
#
#   ./b5_status.sh            # print once
#   ./b5_status.sh 1800       # print every 1800s until the run is done
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
INTERVAL="${1:-0}"
LOG=logs/b5_run.log
TOTAL_STEPS="${MAX_STEPS:-1122}"

status_once() {
    if [[ ! -f $LOG ]]; then echo "[status] no $LOG yet"; return; fi
    local txt arm phase line step spd pct eta loss val
    txt=$(tr '\r' '\n' < "$LOG" | grep -v '^$')

    # Which arm is in flight -- the launcher prints a banner per arm.
    arm=$(echo "$txt" | grep -oE "ARM [AB] — [a-z()-]+" | tail -1)
    [[ -z $arm ]] && arm="ARM ?"
    echo "$txt" | grep -q "ARM B done" && { echo "[status] $(date +%H:%M) | BOTH ARMS DONE"; return; }

    # Most recent tqdm bar tells us the phase and, for training, the progress.
    line=$(echo "$txt" | grep -E "\|.*\| *[0-9]+/[0-9]+ \[" | tail -1)
    phase=$(echo "$line" | sed -E 's/:.*//' | tail -c 40)

    if echo "$line" | grep -q "fine-tune (optimizer steps)"; then
        step=$(echo "$line" | grep -oE "\| [0-9]+/[0-9]+" | grep -oE "[0-9]+" | head -1)
        spd=$(echo "$line" | grep -oE "[0-9.]+s/step" | tail -1)
        eta=$(echo "$line" | grep -oE "<[0-9:]+" | tail -1 | tr -d '<')
        loss=$(echo "$line" | grep -oE "loss=[0-9.]+" | tail -1)
        pct=$(python3 -c "print(f'{100*$step/$TOTAL_STEPS:.1f}')" 2>/dev/null)
        val=$(echo "$txt" | grep -oE "val CE [0-9.]+" | tail -1)
        echo "[status] $(date +%H:%M) | $arm | training ${step}/${TOTAL_STEPS} (${pct}%) | ${spd:-?} | eta ${eta:-?} | ${loss:-} | ${val:-no val yet}"
    else
        step=$(echo "$line" | grep -oE "\| [0-9]+/[0-9]+" | head -1 | tr -d '| ')
        eta=$(echo "$line" | grep -oE "<[0-9:]+" | tail -1 | tr -d '<')
        echo "[status] $(date +%H:%M) | $arm | ${phase:-eval} ${step:-?} | eta ${eta:-?}"
    fi
}

if [[ $INTERVAL -eq 0 ]]; then
    status_once
else
    while true; do
        status_once
        # Stop once the launcher has finished both arms and written the comparison.
        tr '\r' '\n' < "$LOG" 2>/dev/null | grep -q "ARM B done" && break
        pgrep -f "cnndailymail_ddp" > /dev/null || { echo "[status] no training process alive — run ended or died; check $LOG"; break; }
        sleep "$INTERVAL"
    done
fi
