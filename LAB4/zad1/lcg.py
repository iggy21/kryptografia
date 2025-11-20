class LCG:

    def __init__(self, A: int, B: int, m: int, seed: int, n: int = 100):
        if not (0 <= seed < m):
            raise ValueError("Seed musi być z zakresu [0, m).")
        if not (0 < A < m):
            raise ValueError("A musi być z zakresu (0, m).")
        if not (0 <= B < m):
            raise ValueError("B musi być z zakresu [0, m).")
        self.A = A
        self.B = B
        self.m = m
        self.n = n
        self._init_state(seed)

    def _init_state(self, seed: int):
        self.sigma = seed
        self.tau = []
        self.pi = self.n

    def reset(self, seed: int):
        if not (0 <= seed < self.m):
            raise ValueError("Seed musi być z zakresu [0, m).")
        self._init_state(seed)

    def _refresh_state(self):
        self.sigma = (self.A * self.sigma + self.B) % self.m
        bin_str = format(self.sigma, f'0{self.n}b')
        self.tau = [int(b) for b in bin_str]
        self.pi = 0

    def next_bit(self) -> int:
        if self.pi >= self.n or not self.tau:
            self._refresh_state()
        beta = self.tau[self.pi]
        self.pi += 1
        return beta
