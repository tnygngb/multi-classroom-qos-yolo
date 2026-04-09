"""Lightweight centroid tracker for optional post-process association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Track:
    """One naive tracked target."""

    track_id: int
    center_x: float
    center_y: float
    last_seen_frame: int


class NaiveCentroidTracker:
    """
    Very small centroid tracker.

    This tracker is intentionally simple and used as an optional utility.
    It performs nearest-neighbor matching with a distance threshold.
    """

    def __init__(self, *, max_distance: float = 60.0) -> None:
        self.max_distance = float(max_distance)
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, centers: Iterable[tuple[float, float]], *, frame_index: int) -> list[Track]:
        points = [(float(x), float(y)) for x, y in centers]
        assigned_tracks: list[Track] = []

        for center_x, center_y in points:
            track = self._find_best_track(center_x, center_y)
            if track is None:
                track = Track(
                    track_id=self._next_id,
                    center_x=center_x,
                    center_y=center_y,
                    last_seen_frame=int(frame_index),
                )
                self._tracks[track.track_id] = track
                self._next_id += 1
            else:
                track.center_x = center_x
                track.center_y = center_y
                track.last_seen_frame = int(frame_index)
            assigned_tracks.append(track)

        return assigned_tracks

    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def _find_best_track(self, center_x: float, center_y: float) -> Track | None:
        best_track: Track | None = None
        best_distance = self.max_distance

        for track in self._tracks.values():
            dist = ((track.center_x - center_x) ** 2 + (track.center_y - center_y) ** 2) ** 0.5
            if dist <= best_distance:
                best_distance = dist
                best_track = track

        return best_track
