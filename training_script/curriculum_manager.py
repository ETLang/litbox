import torch

class CurriculumManager:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def check_for_graduation(self, current_score):
        # Initialize score
        if self.best_score is None:
            self.best_score = current_score
            return False

        # Check if improvement is significant
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # Reset patience
        else:
            self.counter += 1

        # Level up if patience runs out
        if self.counter >= self.patience:
            self.can_graduate = True
        return self.can_graduate

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.can_graduate = False