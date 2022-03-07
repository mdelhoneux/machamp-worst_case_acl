# Zero-Shot Dependency Parsing with Worst-Case Aware Automated Curriculum Learning

This codebase contains code to reproduce experiments from our [ACL 2022 paper.](https://openreview.net/pdf?id=h0lckggpp4X) (camera-ready version coming soon).
It integrates the method developed by [Zhang et al 2020](https://arxiv.org/pdf/2009.11138.pdf) (which was implemented within the MT-DNN framework) into the [MaChAmp](https://github.com/machamp-nlp/machamp) library. Detailed instructions for how to run models can be found in the machamp repository.
The configurations of our different models are in `config/`. The language configurations can be created with `scripts/create_params.py`, given a UD treebanks location.
The main training script is in `scripts/train_acl22.sh`

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```

@inproceedings{delhoneux2022worst,
  title = "Zero-Shot Dependency Parsing with Worst-Case Aware Automated Curriculum Learning",
  author = "de Lhoneux, Miryam and
  Zhang, Sheng and
  S{\o}gaard, Anders",
  year = "2022",
  url = "https://openreview.net/pdf?id=h0lckggpp4X",
  booktitle = "Proceedings of ACL",
}
```

