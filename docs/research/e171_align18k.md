# E17118k centroid alignment diagnostic

p_aligned=p+mean(GT128)-mean(p); translation only, oracle GT centroid, no rotation/scale or new inference

| Test/method | EMD mm | CD mm | Centroid mm | Coverage5mm % |
|---|---:|---:|---:|---:|
| IID/raw | 11.266 | 6.719 | 8.382 | 45.16 |
| IID/aligned | 7.499 | 5.066 | 0.000 | 61.12 |
| SE3/raw | 312.990 | 291.878 | 312.437 | 1.57 |
| SE3/aligned | 13.857 | 9.150 | 0.000 | 31.10 |

Oracle translation diagnostic, not deployable inference. Shape/size/orientation unchanged. Same stored128points/noise/20Eulersteps.
Code commit `2cfc0bc795efe8456944c3be422e16921f9b34aa`.
