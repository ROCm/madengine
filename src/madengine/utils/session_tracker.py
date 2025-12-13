"""
Session Tracking Utility

Tracks execution sessions to filter current run results from historical data in perf.csv.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import os
from pathlib import Path
from typing import Optional


class SessionTracker:
    """
    Tracks execution session boundaries for filtering performance results.
    
    When an execution starts, it records the current row count in perf.csv.
    After execution, results can be filtered to show only rows added during this session.
    
    Best Practice: Session marker file is stored in the SAME directory as perf.csv
    to ensure consistent access regardless of working directory changes.
    """
    
    def __init__(self, perf_csv_path: str = "perf.csv"):
        """
        Initialize session tracker.
        
        Args:
            perf_csv_path: Path to the performance CSV file
        """
        self.perf_csv_path = Path(perf_csv_path).resolve()  # Use absolute path
        self.session_start_row: Optional[int] = None
        # Marker file in same directory as perf.csv
        self.marker_file = self.perf_csv_path.parent / ".madengine_session_start"
    
    def start_session(self) -> int:
        """
        Mark the start of an execution session.
        
        Records the current number of rows in perf.csv so we can later
        identify which rows were added during this session.
        
        Also saves the marker file for use by child processes.
        
        Returns:
            The starting row number (number of rows in CSV before this session)
        """
        if self.perf_csv_path.exists():
            # Count existing rows (excluding header)
            with open(self.perf_csv_path, 'r') as f:
                lines = f.readlines()
                # Subtract 1 for header row
                self.session_start_row = max(0, len(lines) - 1)
        else:
            # No existing file, start at 0
            self.session_start_row = 0
        
        # Automatically save marker for child processes
        self._save_marker(self.session_start_row)
        
        return self.session_start_row
    
    def get_session_start(self) -> Optional[int]:
        """
        Get the session start row.
        
        Returns:
            Session start row number, or None if session not started
        """
        return self.session_start_row
    
    def get_session_row_count(self) -> int:
        """
        Get the number of rows added during this session.
        
        Returns:
            Number of rows added since session start
        """
        if self.session_start_row is None:
            return 0
        
        if not self.perf_csv_path.exists():
            return 0
        
        with open(self.perf_csv_path, 'r') as f:
            lines = f.readlines()
            current_row_count = max(0, len(lines) - 1)  # Exclude header
        
        return current_row_count - self.session_start_row
    
    def _save_marker(self, start_row: int):
        """
        Save session start marker to file (private method).
        
        Args:
            start_row: The starting row number
        """
        with open(self.marker_file, 'w') as f:
            f.write(str(start_row))
    
    def load_marker(self) -> Optional[int]:
        """
        Load session start marker from file.
        
        Uses the marker file path from this instance's perf_csv_path.
            
        Returns:
            Session start row, or None if file doesn't exist
        """
        if self.marker_file.exists():
            try:
                with open(self.marker_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                return None
        return None
    
    def cleanup_marker(self):
        """
        Remove session marker file for this instance.
        """
        if self.marker_file.exists():
            try:
                os.remove(self.marker_file)
            except OSError:
                pass
    
    @staticmethod
    def load_session_marker_for_csv(perf_csv_path: str = "perf.csv") -> Optional[int]:
        """
        Static helper to load session marker for a given CSV path.
        
        This is useful when you don't have a SessionTracker instance but need to load the marker.
        
        Args:
            perf_csv_path: Path to the performance CSV file
            
        Returns:
            Session start row, or None if marker doesn't exist
        """
        perf_path = Path(perf_csv_path).resolve()
        marker_file = perf_path.parent / ".madengine_session_start"
        
        if marker_file.exists():
            try:
                with open(marker_file, 'r') as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                return None
        return None

