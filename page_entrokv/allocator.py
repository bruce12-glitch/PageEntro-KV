from typing import List, Tuple


def largest_remainder_quotas(b_total: int, L: int, hkv: int, B: int) -> Tuple[List[int], List[int]]:
    base = b_total // (L * hkv * B) if (L * hkv * B) else 0
    quotas = [base] * L
    x = b_total // (hkv * B) if (hkv * B) else 0
    deficit = x - sum(quotas)
    remainders = [0.0] * L
    order = sorted(range(L), key=lambda l: (-remainders[l], l))
    for i in range(max(0, min(deficit, L))):
        quotas[order[i]] += 1
    adjusted = [q for q in quotas]
    return quotas, adjusted
