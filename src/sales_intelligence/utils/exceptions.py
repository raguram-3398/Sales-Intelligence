from __future__ import annotations


class AnalysisError(Exception):
    def __init__(self, message: str, attempts: int, last_error: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
