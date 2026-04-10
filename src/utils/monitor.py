import time
import signal
import sys


class HeartbeatMonitor:
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.last_heartbeat = time.time()
        self.running = True
        
        signal.signal(signal.SIGALRM, self._heartbeat_handler)
        signal.alarm(interval)
    
    def _heartbeat_handler(self, signum, frame):
        elapsed = time.time() - self.last_heartbeat
        print(f"\n[Heartbeat] Training alive, {elapsed:.0f}s since last update", flush=True)
        self.last_heartbeat = time.time()
        if self.running:
            signal.alarm(self.interval)
    
    def stop(self):
        self.running = False
        signal.alarm(0)
    
    def update(self):
        self.last_heartbeat = time.time()


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"
