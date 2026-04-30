# SIGHUP gotcha: why long-running remote work goes under `systemd-run --user`

## The empirical evidence

During the production training run that produced the metrics in §2.7, the FULL training was launched as:

```python
terminal(background=True, command="ssh node-N 'bash launch_full.sh'")
```

The remote python process appeared properly detached (`TT=?`, `STAT=Sl`, multithreaded sleeping, no controlling terminal). It looked safe. Local watcher noise from a `global_step=` watch pattern was firing every few minutes and we wanted to silence it. So we ran:

```python
process(action="kill", session_id="proc_a7f...")
```

The local SSH client died — and within seconds, the remote training also died. GPU dropped to 0%, no python process. **7 minutes of warmup wasted.** Restart was forced under `systemd-run --user --unit=dflash-full --collect bash launch_full.sh`, which has been stable through agent restarts, watcher kills, and SSH session changes ever since.

## Why this happens

When you run `ssh host 'bash launcher.sh'`, the bash you spawn on `host` joins the SSH session group and **stays in it for the lifetime of the connection**. SIGHUP propagation flows:

```
local kill SSH client
  → SSH server-side process exits
    → controlling pty for the remote bash dies
      → kernel sends SIGHUP to the bash
        → bash forwards SIGHUP to its child process group
          → torchrun and its python child both die
```

Plain `nohup`, `disown`, or `setsid` on the launcher are NOT enough — they leave the process in a process group still reachable from the SSH session. They suppress the immediate signal in some configurations, but the specific path through bash → process group → torchrun is robust enough that SIGHUP gets through anyway in practice.

`systemd-run --user --unit=NAME --collect bash launcher.sh` puts the work in **its own transient unit scope** in the user app slice. Path:

```
/user.slice/user-NNNN.slice/user@NNNN.service/app.slice/NAME.service
```

Once it's in there, the SSH session can come and go, the local agent can restart, watchers can be killed — none of it touches the remote work.

## The pattern

### 1. Self-contained remote launcher

```bash
#!/usr/bin/env bash
set -eo pipefail
cd ${WORKSPACE}/repos/speculators
source ${WORKSPACE}/venvs/vllm/bin/activate
exec torchrun ... 2>&1 | tee ${LOG}
```

### 2. Launch via systemd-run user-scope, never bare bash

```bash
ssh host "systemctl --user reset-failed JOBNAME 2>/dev/null; \
  systemd-run --user \
    --unit=JOBNAME \
    --description='human-readable description' \
    --collect \
    bash /path/to/launcher.sh"
```

`--collect` auto-cleans the unit when it exits.

### 3. Verify it landed in user-scope, not session-scope

```bash
ssh host "systemctl --user status JOBNAME --no-pager | head -20"
```

Look for the CGroup line. It MUST contain `app.slice/JOBNAME.service`. If instead it shows `session-NNN.scope`, you skipped systemd-run and the process is still SSH-bound — fix it before trusting the run.

### 4. Now safe to also tail with a local watcher

A separate `terminal(background=True, command="ssh host 'tail -f log'")` watcher can be killed at any time without affecting the remote work. Reattach later with another fresh `ssh host 'tail -f log'`.

### 5. Stopping cleanly

```bash
ssh host "systemctl --user stop JOBNAME"
```

## When NOT to migrate an already-running process

If a bare `ssh host 'bash launcher.sh'` background process is already running and producing valuable progress, **do NOT** try to migrate it mid-flight — the migration requires killing it. Just leave it alone, accept the constraint that you can't kill its local watcher safely, and let it finish naturally. Apply this pattern on the *next* launch.

## Pitfalls

- **`systemctl --user` requires a logged-in user manager.** Most modern distros enable lingering-on-boot per user; if your distro doesn't, run `loginctl enable-linger USER` once for the training user.
- **Pick a unique unit name per run** (timestamp suffix is fine). The `reset-failed` is cheap insurance against a name collision from a previous run that died.
- **Logs**: prefer `tee /path/to/file.log` inside the launcher rather than relying on journald, so you have a flat file for `tail` / `grep` without journalctl pagination.
- **`--user` not system**: avoids needing root.

## Quick verification

To confirm the work is in its own scope:

```bash
ssh host "systemctl --user list-units 'JOBNAME*' --no-pager"
ssh host "systemctl --user show JOBNAME --property=ControlGroup,SubState"
```

Expect `SubState=running` and a `ControlGroup` ending in `JOBNAME.service`.
