# 🚨 NEVER TOUCH SPARK-5 🚨

## The rule

**Do not, under any circumstance, do any of the following on spark-5:**

- `ssh spark-5 'pkill ...'`
- `ssh spark-5 'systemctl --user stop ...'`
- `ssh spark-5 '<anything that allocates GPU>'`
- `ssh spark-5 '<anything that allocates large RAM>'`
- `ssh spark-5 '<anything that writes to /home/user or any user dir>'`
- `rsync ... operator@192.168.200.5:...`
- Launch any tmux session, systemd unit, or background job on spark-5
- Modify any file on spark-5
- Touch any process owned by `baby` or any other user on spark-5

## What spark-5 IS for

**Jumphost only.** spark-1's Tailscale has been offline 22+ days. The only way to reach spark-1 is `ssh -J operator@100.66.198.32 operator@10.0.0.103` where `100.66.198.32` is spark-5. Use spark-5 EXCLUSIVELY as that ProxyJump target. Don't run anything on it directly. Don't evict anyone's processes on it. Don't touch.

## What if I think I need spark-5?

You don't. Pick from spark-2/3/4/6 instead. Or queue the work for later. Or ask the user explicitly with the words "I'd like to break the never-touch-spark-5 rule because X" — and then do not break it without an explicit "yes, do it" reply.

## Why this rule exists

User issued it as a hard rule on 2026-04-30. It's older and more important than every other rule in the project doctrine. Violating it is a higher-severity error than corrupting traces, killing a training run, or losing checkpoints.

## Acknowledged

- Memory entry (replicated): "HARD RULE: NEVER touch spark-5. Do not evict, kill processes on, run workloads on, or modify state on spark-5 under any circumstance."
- This file
- Banner at the top of `repro/plan/00-resumability-doctrine.md`
- Banner at the top of `repro/plan/2026-04-30-iq4-worker-orchestration.md`
- Banner in `repro/plan/00-INDEX.md`
- Inline reminder in the orchestration doc's "Add a future worker" section
- Inline reminder in the "Pool traces" recipe (`for src in ...; do` excludes spark-5)
