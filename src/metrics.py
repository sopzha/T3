Q_ERROR_CUTOFF = 1e-10


def q_error(real: float, estimate: float) -> float:
    assert real >= 0, f"real runtime should be >= 0 but is {real}"
    assert estimate >= 0
    if real <= Q_ERROR_CUTOFF:
        real = Q_ERROR_CUTOFF
    if estimate <= Q_ERROR_CUTOFF:
        estimate = Q_ERROR_CUTOFF
    return max(real / estimate, estimate / real)


def abs_error(real: float, estimate: float) -> float:
    return max(real - estimate, estimate - real)
