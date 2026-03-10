import numpy as np

class ConfidenceEngine:

    def __init__(self, threshold=0.65, modality_weights=None):

        self.threshold = threshold

        self.modality_weights = modality_weights or {
            "text": 1.0,
            "image": 0.7,
            "audio": 0.6,
            "video": 0.8
        }

    def compute_confidence(self, similarity_scores):

        weighted_sum = 0.0
        weight_total = 0.0

        for modality, score in similarity_scores.items():

            if modality not in self.modality_weights:
                continue

            if score is None:
                continue

            weight = self.modality_weights[modality]

            weighted_sum += score * weight
            weight_total += weight

        if weight_total == 0:
            return {
                "confidence": 0.0,
                "is_match": False
            }

        confidence = weighted_sum / weight_total

        return {
            "confidence": float(confidence),
            "is_match": confidence >= self.threshold
        }