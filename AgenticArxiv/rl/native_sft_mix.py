"""A single deterministic epoch with complete coverage and a 60:30:10 decision mix."""
import math
import random

WEIGHTS = {"core":6,"error_targeted":3,"special":1}


def epoch_indices(buckets, seed=42):
    pools={k:[] for k in WEIGHTS}
    for i,k in enumerate(buckets):
        pools[k].append(i)
    if not all(pools.values()):
        raise ValueError("Missing SFT sampling bucket")
    units=max(math.ceil(len(pools[k])/weight) for k,weight in WEIGHTS.items())
    rng=random.Random(seed)
    order=[]
    for k,weight in WEIGHTS.items():
        count=units*weight
        pool=pools[k].copy()
        while count:
            rng.shuffle(pool)
            take=min(count,len(pool))
            order.extend(pool[:take])
            count-=take
    rng.shuffle(order)
    return order
