"""Subprocess helpers for tracegen server/client lifecycle."""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from typing import Callable, Optional

_PR_SET_PDEATHSIG = 1


def parent_deathsig_preexec() -> Optional[Callable[[], None]]:
    """Return a Linux-only pre-exec hook that SIGKILLs children on parent death."""
    if sys.platform != "linux":
        return None

    libc = ctypes.CDLL(None, use_errno=True)

    def _set_parent_deathsig() -> None:
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        # If the parent already died before prctl() ran, self-terminate so the
        # child does not stay resident and wedge retries.
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)

    return _set_parent_deathsig
