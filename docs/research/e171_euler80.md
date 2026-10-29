# E171last18k — Euler20/40/80

Same512IIDqueries,noise167,weights andsource; fullFM interval[0,1],dt=1/N. No training or GTalignment.

| Eulersteps | Centroid mm ↓ | EMD mm ↓ | CD mm ↓ | AlignedCD mm ↓ | Coverage5mm % ↑ |
|---:|---:|---:|---:|---:|---:|
| 20 | 8.382 | 11.266 | 6.719 | 5.066 | 45.16 |
| 40 | 8.712 | 11.530 | 6.867 | 5.069 | 43.39 |
| 80 | 8.755 | 11.709 | 6.980 | 5.203 | 42.85 |

Baseline20 endpoint reproduced bit-exact, EMD/coverage checked. Single checkpoint/fixednoise; no inference beyond t=1.
Source commit `e4d18c231ae8fbfcb47d7893230476fedb44f4f0`.
