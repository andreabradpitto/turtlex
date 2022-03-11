import random
import numpy as np
from collections import deque


class SumTree:
    """
    A binary tree data structure where the parent’s
    value is the sum of its children
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # Update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # Find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # Store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # Get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class MemoryBuffer(object):
    """
    Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False):
        """
        Initialization
        """
        if(with_per):
            # Prioritized Experience Replay (propositional)
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, error=None):
        """
        Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, action, reward, done, new_state)
        if(self.with_per):
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """
        Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """
        Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() / batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch], dtype="object")
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch], dtype="object")
        a_batch = np.array([i[1] for i in batch], dtype="object")
        r_batch = np.array([i[2] for i in batch], dtype="object")
        d_batch = np.array([i[3] for i in batch], dtype="object")
        new_s_batch = np.array([i[4] for i in batch], dtype="object")
        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def update(self, idx, new_error):
        """
        Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """
        Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(self.buffer_size)
        else: self.buffer = deque()
        self.count = 0


class OrnsteinUhlenbeckProcess(object):
    """
    Ornstein-Uhlenbeck Noise
    """
    def __init__(self, theta=0.15, mu=0, sigma=0.2, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x
