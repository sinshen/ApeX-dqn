# Ape-X dqn

### What deepmind said is right, prioritization is the most important ingredient contributing to the agent's performance.

You can run ape-x dqn by
```
python apexdqn.py
```

I use distribution value, for more information in [c51-qr-dqn](https://github.com/LihaoR/c51-qr-dqn)

And I use ```threading.Lock``` for shared memory, it makes my code slow. But my apexdqn will also converge much faster than vanilla async-qr-dqn, a3c...

## To do

Something wrong in Prioritised Experience Replay, but it doesn't matter the convergence of this algorithm.

Since I can't run it with 360 actors, so I don't know if it can perform so good as the paper said

## paper

[Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)
