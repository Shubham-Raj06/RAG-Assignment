import sys
import json
import time

class StructuredLogger:
    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, message: str, **kwargs):
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "logger": self.name,
            "level": level,
            "message": message,
        }
        if kwargs:
            payload.update(kwargs)
        # Emit logs to stderr to keep stdout perfectly clean for CLI parsing
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)
