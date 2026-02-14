# src/mht_tracker.py
import numpy as np
import copy
from dataclasses import dataclass
from typing import List
from scipy.optimize import linear_sum_assignment


@dataclass
class Hypothesis:
    tracks: list
    log_score: float
    history: list


class MHTTracker:
    def __init__(self,
                 gating_distance=14.0,
                 max_hypotheses=20,
                 n_scan=3,
                 max_missed=10,
                 miss_penalty=6.0,
                 birth_penalty=8.0):
        self.gating_distance = gating_distance
        self.max_hypotheses = max_hypotheses
        self.n_scan = n_scan
        self.max_missed = max_missed
        self.miss_penalty = miss_penalty
        self.birth_penalty = birth_penalty
        self.hypotheses: List[Hypothesis] = []

    def initialize(self, tracks):
        self.hypotheses = [
            Hypothesis(
                tracks=copy.deepcopy(tracks),
                log_score=0.0,
                history=[]
            )
        ]

    def _predict_tracks(self, tracks):
        for tr in tracks:
            tr.kf.predict()
            tr.missed += 1
            tr.total_predictions += 1

    def _build_cost(self, tracks, detections):
        if not tracks or not detections:
            return None

        cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float32)
        for i, tr in enumerate(tracks):
            pred = tr.kf.x[:2, 0]
            for j, det in enumerate(detections):
                cx = det[0] + det[2] / 2
                cy = det[1] + det[3] / 2
                d = np.linalg.norm(pred - np.array([cx, cy]))
                if d <= self.gating_distance:
                    cost[i, j] = d
        return cost

    def step(self, detections, frame_id, next_id, create_track_fn):
        new_hyps = []

        for hyp in self.hypotheses:
            tracks = copy.deepcopy(hyp.tracks)

            # ===== 1️⃣ 预测 =====
            self._predict_tracks(tracks)

            cost = self._build_cost(tracks, detections)
            matched_det = set()
            log_score = hyp.log_score

            if cost is not None and cost.size > 0:
                row, col = linear_sum_assignment(cost)

                for r, c in zip(row, col):
                    if cost[r, c] >= 1e5:
                        continue
                    det = detections[c]
                    cx = det[0] + det[2] / 2
                    cy = det[1] + det[3] / 2

                    tracks[r].kf.update(np.array([cx, cy]))
                    tracks[r].missed = 0
                    tracks[r].hits += 1
                    tracks[r].last_update_frame = frame_id
                    tracks[r].total_updates += 1

                    log_score -= cost[r, c]
                    matched_det.add(c)

            # ===== 2️⃣ 未匹配检测 → 新生轨迹 =====
            for i, det in enumerate(detections):
                if i in matched_det:
                    continue
                tr = create_track_fn(next_id, frame_id, det)
                tracks.append(tr)
                next_id += 1
                log_score -= self.birth_penalty

            # ===== 3️⃣ 删除超时轨迹 =====
            tracks = [tr for tr in tracks if tr.missed <= self.max_missed]

            new_hyps.append(
                Hypothesis(
                    tracks=tracks,
                    log_score=log_score,
                    history=hyp.history + [(frame_id, len(tracks))]
                )
            )

        # ===== 4️⃣ 剪枝 =====
        new_hyps.sort(key=lambda h: h.log_score, reverse=True)
        self.hypotheses = new_hyps[:self.max_hypotheses]

        # ===== 5️⃣ N-scan 剪枝 =====
        if len(self.hypotheses) > 1 and len(self.hypotheses[0].history) > self.n_scan:
            ref = self.hypotheses[0].history[:-self.n_scan]
            self.hypotheses = [
                h for h in self.hypotheses
                if h.history[:-self.n_scan] == ref
            ]

        best = self.hypotheses[0]
        return best.tracks, next_id
