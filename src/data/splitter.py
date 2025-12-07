from typing import Iterator, Tuple

class ExpandingWindowSplitter:
    """
    Generates (train_start, train_end), (val_start, val_end), (test_start, test_end)
    indices for expanding window backtesting.
    """
    def __init__(
        self, 
        total_samples: int, 
        initial_train_size: int, 
        test_size: int, 
        gap: int = 0
    ):
        self.total = total_samples
        self.train_size = initial_train_size
        self.test_size = test_size
        self.gap = gap # Gap between train and test to prevent leakage (optional)

    def split(self) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        Yields:
            (train_idx_range, val_idx_range, test_idx_range)
            Each is a tuple (start, end)
        """
        # We'll use the 'test_size' as the validation size as well for simplicity
        val_size = self.test_size
        
        current_train_end = self.train_size
        
        fold = 0
        while current_train_end + val_size + self.test_size <= self.total:
            # Train: [0, current_train_end)
            train_range = (0, current_train_end)
            
            # Val: [current_train_end, current_train_end + val_size)
            val_start = current_train_end
            val_end = val_start + val_size
            val_range = (val_start, val_end)
            
            # Test: [val_end, val_end + test_size)
            test_start = val_end
            test_end = test_start + self.test_size
            test_range = (test_start, test_end)
            
            yield train_range, val_range, test_range
            
            # Move forward
            current_train_end += self.test_size # Expand training window
            fold += 1