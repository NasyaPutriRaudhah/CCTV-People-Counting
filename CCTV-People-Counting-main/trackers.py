import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from config import MAX_TRACKING_DISTANCE


class CentroidLineCrossingTracker:
    """
    Tracker dengan 1 garis crossing.
    - Gerak ke BAWAH melewati garis → ENTRY (+1)
    - Gerak ke ATAS melewati garis   → EXIT  (-1)
    """

    MIN_FRAMES_BEFORE_CROSSING = 3

    def __init__(self, line_y, max_disappeared=10, max_distance=MAX_TRACKING_DISTANCE):
        self.line_y = line_y
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # OrderedDict: {object_id: {cx, cy, disappeared, has_crossed, frame_count}}
        self.objects = OrderedDict()
        self.next_object_id = 0

    # ------------------------------------------------------------------
    # Register / Deregister
    # ------------------------------------------------------------------

    def _register(self, cx, cy):
        self.objects[self.next_object_id] = {
            'cx': cx,
            'cy': cy,
            'disappeared': 0,
            'has_crossed': False,   # sudah pernah crossing → jangan hitung 2x
            'frame_count': 0,
        }
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, detections):
        """
        Parameters
        ----------
        detections : list of dict dengan key 'cx' dan 'cy'

        Returns
        -------
        entries : int
        exits   : int
        """
        entries = 0
        exits = 0

        # Tidak ada deteksi → semua track jadi disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    self._deregister(object_id)
            return entries, exits

        # Belum ada track → register semua
        if len(self.objects) == 0:
            for det in detections:
                self._register(det['cx'], det['cy'])
            return entries, exits

        # Ada track & ada deteksi → matching dengan distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = np.array([[o['cx'], o['cy']]
                                      for o in self.objects.values()])
        input_centroids = np.array([[d['cx'], d['cy']] for d in detections])

        D = dist.cdist(object_centroids, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            track = self.objects[object_id]

            new_cx = input_centroids[col][0]
            new_cy = input_centroids[col][1]
            prev_cy = track['cy']

            track['frame_count'] += 1

            # Cek crossing hanya kalau track sudah cukup tua
            if track['frame_count'] >= self.MIN_FRAMES_BEFORE_CROSSING:
                entries, exits = self._check_crossing(
                    track, object_id, new_cx, new_cy, prev_cy, entries, exits
                )

            # Update posisi kalau track belum dideregister
            if object_id in self.objects:
                self.objects[object_id]['cx'] = new_cx
                self.objects[object_id]['cy'] = new_cy
                self.objects[object_id]['disappeared'] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Track yang tidak dapat pasangan → disappeared
        for row in set(range(len(object_ids))) - used_rows:
            object_id = object_ids[row]
            self.objects[object_id]['disappeared'] += 1
            if self.objects[object_id]['disappeared'] > self.max_disappeared:
                self._deregister(object_id)

        # Deteksi baru yang tidak match track manapun → register
        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col][0], input_centroids[col][1])

        return entries, exits

    # ------------------------------------------------------------------
    # Logika crossing — 1 garis, 2 arah
    # ------------------------------------------------------------------

    def _check_crossing(self, track, object_id, new_cx, new_cy, prev_cy,
                         entries, exits):
        """
        Satu garis (self.line_y):
        - prev_cy < line_y AND new_cy >= line_y → gerak ke BAWAH → ENTRY
        - prev_cy > line_y AND new_cy <= line_y → gerak ke ATAS  → EXIT
        has_crossed mencegah double count kalau orang mondar-mandir di dekat garis
        """
        crossed = track.get('has_crossed', False)

        # Gerak ke BAWAH → ENTRY
        if prev_cy > self.line_y and new_cy <= self.line_y:
            if not crossed:
                entries += 1
                print(f"✓ [ENTRY +1] ID={object_id} "
                      f"(y: {prev_cy:.0f}→{new_cy:.0f}, line_y={self.line_y})")
                if object_id in self.objects:
                    self._deregister(object_id)
            else:
                print(f"⊘ [SKIP ENTRY] ID={object_id} sudah pernah crossing")
                if object_id in self.objects:
                    self.objects[object_id]['has_crossed'] = True

        # Gerak ke ATAS → EXIT
        elif prev_cy < self.line_y and new_cy >= self.line_y:
            if not crossed:
                exits += 1
                print(f"✓ [EXIT -1] ID={object_id} "
                      f"(y: {prev_cy:.0f}→{new_cy:.0f}, line_y={self.line_y})")
                if object_id in self.objects:
                    self._deregister(object_id)
            else:
                print(f"⊘ [SKIP EXIT] ID={object_id} sudah pernah crossing")
                if object_id in self.objects:
                    self.objects[object_id]['has_crossed'] = True

        return entries, exits

    # ------------------------------------------------------------------

    def get_status_info(self):
        return {
            'active_tracks': len(self.objects),
            'track_ids': list(self.objects.keys()),
        }
