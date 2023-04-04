import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 163596104 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    s = np.sum(x) / 80
    var = np.sum((x / 80 - s) ** 2) / (n - 1)
    t = norm.ppf(1 - alpha / 2)
    left = s - t * np.sqrt(var / n) * np.exp(-1 / 2) / np.sqrt(2 * (n - 1))
    right = s + t * np.sqrt(var / n) * np.exp(-1 / 2) / np.sqrt(2 * (n - 1))
    return left, right
