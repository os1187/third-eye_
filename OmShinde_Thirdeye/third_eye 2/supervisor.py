import logging
import threading
import time
import traceback
import random
from typing import Callable, Optional, Any, Dict, Tuple

# new imports for JSON logging and rotation
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class Supervisor:
    """
    A small thread supervisor utility providing:
    - Start with retries and exponential backoff
    - Periodic aliveness checks via a monitor thread
    - Structured logging on exceptions and lifecycle events
    - Graceful shutdown with join timeouts

    Usage:
        sup = Supervisor(
            name="audio-worker",
            target=worker_fn,
            args=(...),
            kwargs={...},
        )
        started = sup.start(retries=3, backoff_initial_s=0.05, backoff_factor=2.0)
        # ... later ...
        sup.stop(timeout_s=1.0)
    """

    def __init__(
        self,
        name: str,
        target: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        check_interval_s: float = 0.5,
        restart_on_failure: bool = False,
    ) -> None:
        self.name = name
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._logger = logger or logging.getLogger(f"supervisor.{name}")
        self._check_interval_s = max(0.05, check_interval_s)
        self._restart_on_failure = restart_on_failure

        self._stop_event = threading.Event()
        # Ensure target can observe stop request if it supports it
        self._kwargs.setdefault("stop_event", self._stop_event)

        self._thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_exception: Optional[BaseException] = None
        self._attempts = 0

    # ---------------------- public API ----------------------
    @property
    def thread(self) -> Optional[threading.Thread]:
        return self._thread

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_exception(self) -> Optional[BaseException]:
        return self._last_exception

    def start(
        self,
        retries: int = 3,
        backoff_initial_s: float = 0.05,
        backoff_factor: float = 2.0,
        backoff_max_s: float = 2.0,
        boot_grace_s: float = 0.05,
    ) -> bool:
        """
        Attempt to start the worker, retrying on early failure with backoff.
        Returns True if started and alive, False otherwise.
        """
        self._stop_event.clear()
        self._attempts = 0
        attempt = 0
        backoff = max(0.0, backoff_initial_s)

        while attempt <= max(0, retries):
            self._attempts = attempt + 1
            self._last_exception = None

            # Wrap target to capture exceptions
            def _run_wrapper():
                try:
                    self._target(*self._args, **self._kwargs)
                except BaseException as e:  # capture all exceptions
                    self._last_exception = e
                    self._log(
                        "thread_exception",
                        error=str(e),
                        traceback=traceback.format_exc(),
                        attempt=self._attempts,
                    )
                finally:
                    # lifecycle event: thread exiting
                    self._log("thread_exit", attempt=self._attempts)

            self._thread = threading.Thread(target=_run_wrapper, name=f"{self.name}-worker", daemon=True)
            self._thread.start()
            self._log("thread_start", attempt=self._attempts)

            # Give the worker a short boot grace period to crash if it's going to
            time.sleep(max(0.0, boot_grace_s))
            if self._thread.is_alive():
                # Consider started; spawn monitor
                self._running = True
                self._start_monitor()
                return True

            # Early failure; apply backoff and retry
            self._log(
                "start_failed",
                backoff_s=backoff,
                attempt=self._attempts,
                error=(str(self._last_exception) if self._last_exception else None),
            )
            attempt += 1
            if attempt > retries:
                break
            # add small jitter to reduce thundering herd
            jitter = random.uniform(0.0, min(0.01, backoff / 4.0))
            time.sleep(backoff + jitter)
            backoff = min(backoff * max(1.0, backoff_factor), backoff_max_s)

        return False

    def stop(self, timeout_s: float = 1.0) -> bool:
        """
        Request the worker to stop and attempt to join within timeout.
        Returns True if joined cleanly, False if timed out.
        """
        self._stop_event.set()
        joined = True

        if self._thread:
            self._thread.join(timeout=max(0.0, timeout_s))
            if self._thread.is_alive():
                joined = False
                self._log("stop_timeout", timeout_s=timeout_s)
            else:
                self._log("stopped")

        # stop monitor
        if self._monitor_thread:
            # monitor reads stop_event; it will exit on next cycle
            self._monitor_thread.join(timeout=max(0.0, timeout_s))

        self._running = False
        return joined

    # ---------------------- internals ----------------------
    def _start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        def _monitor_loop():
            while not self._stop_event.is_set():
                time.sleep(self._check_interval_s)
                if not self._thread:
                    continue
                if not self._thread.is_alive():
                    self._log("thread_dead")
                    if not self._stop_event.is_set() and self._restart_on_failure:
                        # Optional runtime restart, one attempt with current backoff of 0 (fast)
                        self._log("runtime_restart")
                        self.start(retries=0, backoff_initial_s=0.0, boot_grace_s=self._check_interval_s)
                        # If it fails, the next monitor cycle will log again
                        break

        self._monitor_thread = threading.Thread(
            target=_monitor_loop, name=f"{self.name}-monitor", daemon=True
        )
        self._monitor_thread.start()
        self._log("monitor_start")

    def _log(self, event: str, **fields: Any) -> None:
        payload = {
            "action": event,
            "event": event,  # backward compatibility for existing tests/consumers
            "thread": threading.current_thread().name,
            "ts": time.time(),
            "name": self.name,
            **fields,
            "context": {"name": self.name, **fields},
        }
        try:
            # Emit structured log with payload attached for consumers
            self._logger.info("", extra={"payload": payload})
        except Exception:
            # Fallback plain logging to avoid raising within logging
            self._logger.info(json.dumps(payload))


# ---------------------- logging utilities ----------------------
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {
                "action": record.getMessage(),
                "thread": threading.current_thread().name,
                "ts": time.time(),
                "context": {},
            }
        return json.dumps(payload, ensure_ascii=False)


class SizeGuardRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename: str, maxBytes: int, backupCount: int, max_total_bytes: Optional[int] = None):
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount)
        self.max_total_bytes = max_total_bytes

    def _log_files(self) -> list[str]:
        files = [self.baseFilename]
        for i in range(1, self.backupCount + 1):
            files.append(f"{self.baseFilename}.{i}")
        return [f for f in files if os.path.exists(f)]

    def emit(self, record: logging.LogRecord) -> None:
        # prune oldest backups if total size exceeds guard
        if self.max_total_bytes:
            files = self._log_files()
            total = sum(os.path.getsize(f) for f in files)
            while total > self.max_total_bytes and len(files) > 1:
                oldest = sorted(files, key=lambda f: os.path.getmtime(f))[0]
                try:
                    os.remove(oldest)
                except Exception:
                    pass
                files = self._log_files()
                total = sum(os.path.getsize(f) for f in files)
        super().emit(record)


def configure_json_file_logger(
    logger: logging.Logger,
    log_path: str,
    rotate: str = "size",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    when: str = "midnight",
    interval: int = 1,
    utc: bool = True,
    max_total_bytes: Optional[int] = None,
) -> logging.Handler:
    """
    Configure a logger to write JSON lines to a rotating file.
    - rotate="size": uses RotatingFileHandler(max_bytes, backup_count)
    - rotate="time": uses TimedRotatingFileHandler(when, interval, backup_count)
    - Each record is emitted as a single JSON object with keys: action, thread, ts, context
    - max_total_bytes: optional guard to prune oldest backups when cumulative size exceeds this threshold
    """
    # remove existing file handlers to avoid duplicates
    for h in list(logger.handlers):
        if isinstance(h, (RotatingFileHandler, TimedRotatingFileHandler, SizeGuardRotatingFileHandler)):
            logger.removeHandler(h)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if rotate == "time":
        handler = TimedRotatingFileHandler(log_path, when=when, interval=interval, backupCount=backup_count, utc=utc)
    else:
        handler = SizeGuardRotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, max_total_bytes=max_total_bytes)

    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return handler