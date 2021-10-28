# LogME
This is the codebase for the following two papers:

- [LogME: Practical Assessment of Pre-trained Models for Transfer Learning](http://proceedings.mlr.press/v139/you21b.html), ICML 2021

- [Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs](https://arxiv.org/abs/2110.10545), arxiv 2021

**Note**: the second paper is an extended version of the first conference paper.

# How to use

## Use LogME to assess transferability

The API looks like sci-kit learn: first initialize an object, and then fit it to your data to get the transferability metric.

By fitting the features ``f`` and labels ``y``, and you can get a nice score which well correlates with the transfer learning performance (without hyper-parameter tuning).

```python
from LogME import LogME
logme = LogME(regression=False)
# f has shape of [N, D], y has shape [N]
score = logme.fit(f, y)
```

Then you can use the ``score``  to quickly select a good pre-trained model. The larger the ``score`` is,  the better transfer performance you get.

After fitting old data, logme can also be used to make prediction on new data:

```python
# f_test has shape of [N_test, D], prediction has shape [N]
prediction = logme.predict(f_test)
```

Meanwhile, the LogME score can also be used to purely measure the compatibility/transferability between features and labels, just like [this paper](https://arxiv.org/abs/2109.01087) from UC Berkeley. 

## Use B-Tuning to fine-tune with multiple (heterogeneous) pre-trained models

This is the second step in the proposed "ranking and tuning" paradigm. Code is coming soon.

# Code for LEEP and NCE

We have received several requests for the code of LEEP and NCE, therefore we release the code in this repository to help the community.

Please see the LEEP.py and NCE.py for details.

Note that LEEP and NCE requires predictions over the pre-trained classes as input. The typical usage may look like:

```python
# get the prediction of shape [N, C_s] from the pre-trained model
# N is the number of samples, C_s is the number of pre-trained classes
import numpy as np
from LEEP import LEEP
from NCE import NCE

pseudo_source_label = xxx
target_label = xxx  # target_label has shape of [N], with its elements in [0, C_t)

leep_score = LEEP(pseudo_source_label, target_label)
nce_score = NCE(np.argmax(pseudo_source_label, axis=1), target_label)
```

# Citation

If you find the code useful, please cite the following papers:

```
@inproceedings{you_logme:_2021,
	title = {LogME: Practical Assessment of Pre-trained Models for Transfer Learning},
	booktitle = {ICML},
	author = {You, Kaichao and Liu, Yong and Wang, Jianmin and Long, Mingsheng},
	year = {2021}
}

@article{you_ranking_2021,
	title = {Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs},
	journal = {arXiv:2110.10545 [cs]},
	author = {You, Kaichao and Liu, Yong and Wang, Jianmin and Jordan, Michael I. and Long, Mingsheng},
	year = {2021}
}
```

# Contact

If you have any question or want to use the code, please contact youkaichao@gmail.com .