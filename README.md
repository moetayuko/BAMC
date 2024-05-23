# Bidirectional Attentive Multi-View Clustering

This is the official implementation of our TKDE paper _Bidirectional Attentive
Multi-View Clustering_.

```bibtex
@ARTICLE{10243106,
  author={Lu, Jitao and Nie, Feiping and Dong, Xia and Wang, Rong and Li, Xuelong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  title={Bidirectional Attentive Multi-View Clustering},
  year={2024},
  volume={36},
  number={5},
  pages={1889-1901},
  keywords={Space exploration;Symmetric matrices;Optimization;Clustering algorithms;Tuning;Terminology;Probabilistic logic;Multi-view clustering;bidirectional attentive clustering;structured graph learning},
  doi={10.1109/TKDE.2023.3312794}}
```

Some parts are implemented in C++ so it's necessary to compile them before the first
run. Make sure you have MATLAB, CMake, and a compatible C++ compiler installed, then
execute the following commands to compile:

```bash
cd funs/EProjSimplex
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

After compilation succeed, you can run `BAMC_demo.m` in MATLAB as usual.
