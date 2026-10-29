# E171last18k — Centroid error across20Euler steps

Raw, unaligned prediction.128IIDscenes×4tasks=512queries; same checkpoint18000/noise167/FP32/Euler20. Step0 is Gaussian source, step20 is t=1 output. GT used only for error measurement. Endpoint reproduces saved evaluation exactly (maxpointdrift0mm).

| Completed steps | Mean mm | Median mm | Q25 mm | Q75 mm |
|---:|---:|---:|---:|---:|
| 0 | 342.381 | 361.971 | 262.912 | 419.443 |
| 1 | 324.918 | 343.481 | 249.236 | 398.066 |
| 2 | 307.585 | 325.085 | 235.585 | 376.923 |
| 3 | 290.308 | 306.769 | 222.282 | 355.853 |
| 4 | 273.093 | 288.432 | 209.199 | 334.925 |
| 5 | 255.885 | 270.114 | 196.293 | 313.927 |
| 6 | 238.698 | 251.890 | 182.824 | 292.809 |
| 7 | 221.411 | 233.747 | 169.317 | 271.983 |
| 8 | 204.071 | 215.606 | 155.437 | 250.978 |
| 9 | 186.834 | 197.349 | 142.185 | 229.979 |
| 10 | 169.798 | 179.564 | 129.258 | 209.371 |
| 11 | 152.736 | 161.507 | 116.226 | 188.253 |
| 12 | 135.488 | 143.263 | 102.699 | 167.381 |
| 13 | 118.201 | 124.677 | 89.880 | 145.887 |
| 14 | 101.265 | 106.571 | 77.457 | 125.155 |
| 15 | 84.278 | 88.352 | 64.452 | 104.626 |
| 16 | 67.421 | 70.501 | 51.407 | 83.521 |
| 17 | 50.358 | 52.314 | 38.591 | 62.415 |
| 18 | 33.073 | 33.536 | 25.262 | 41.342 |
| 19 | 16.891 | 16.596 | 12.065 | 22.055 |
| 20 | 8.382 | 7.824 | 5.504 | 10.559 |

Mean error decreases342.381→8.382mm; step18/19/20 means33.073/16.891/8.382mm. Aggregate curve shows no late increase. This is a trajectory over t∈[0,1], not a test of increasing sampler resolution; one cannot infer that merely adding steps after20 would fix residual translation.

[PNG](/workspace/robotics_assets/experiments/e171_nose3/test_centroid20/results/centroid_error.png) · [Interactive](http://localhost:17070/e171_centroid20.html) · [CSV](/workspace/robotics_assets/experiments/e171_nose3/test_centroid20/results/centroid_error.csv)

Own source commit `c12109556acac906d3c0939e493da133a320032c`; HTTP200,4827296bytes, PNGvisuallyinspected. E172 remains paused at user request.
