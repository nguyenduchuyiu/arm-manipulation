"""Cover all 24 object permutations in each full minibatch."""
import numpy as np


def combinatorial_batches(plans, epochs, batch_size, rng):
    samples, groups = [], []
    offset = 0
    for cluster in plans:
        if len(cluster) != 96 or batch_size < 24:
            raise ValueError("24 permutations x4 queries and batch>=24 required")
        queues = [[] for _ in range(24)]
        for episode, plan in enumerate(cluster):
            for chunk, active in plan:
                if not active:
                    raise ValueError("pick pretraining contains expert chunks only")
                queues[episode // 4].append(len(samples))
                samples.append((offset + episode, chunk, True))
        count = sum(map(len, queues))
        if min(map(len, queues)) < count // batch_size:
            raise ValueError("batch too small for full permutation coverage without resampling")
        groups.append(queues)
        offset += 96
    batches, epoch_ends = [], []
    for _ in range(epochs):
        for group_index in rng.permutation(len(groups)):
            queues = [rng.permutation(items).tolist() for items in groups[group_index]]
            full_batches = sum(map(len, queues)) // batch_size
            for index in range(full_batches):
                batch = [queue.pop() for queue in queues]
                reserve = full_batches - index - 1
                # Reserve one item per permutation for every later full batch.
                available = [(permutation, item) for permutation, queue in enumerate(queues)
                             for item in queue[:len(queue) - reserve]]
                chosen = rng.choice(len(available), batch_size - 24, replace=False)
                used = [set() for _ in range(24)]
                for choice in chosen:
                    permutation, item = available[choice]
                    batch.append(item); used[permutation].add(item)
                queues = [[item for item in queue if item not in used[p]] for p, queue in enumerate(queues)]
                rng.shuffle(batch)
                batches.append(batch)
            remaining = [item for queue in queues for item in queue]
            if remaining:
                rng.shuffle(remaining)
                batches.append(remaining)
        epoch_ends.append(len(batches))
    return samples, batches, epoch_ends
