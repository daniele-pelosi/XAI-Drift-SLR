# TSUNAMI - an explainable PPM approach for customer churn prediction in evolving retail data environments

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Annalisa Appice, Giuseppe Ieva, Donato Malerba*

[*TSUNAMI - an explainable PPM approach for customer churn prediction in evolving retail data environments*](https://link.springer.com/article/10.1007/s10844-023-00838-5)

```
@Article{Pasquadibisceglie2023,
author={Pasquadibisceglie, Vincenzo
and Appice, Annalisa
and Ieva, Giuseppe
and Malerba, Donato},
title={TSUNAMI - an explainable PPM approach for customer churn prediction in evolving retail data environments},
journal={Journal of Intelligent Information Systems},
year={2023},
month={Dec},
day={28},
issn={1573-7675},
doi={10.1007/s10844-023-00838-5},
url={https://doi.org/10.1007/s10844-023-00838-5}
}
```

# How to use:

TSUNAMI requires the following parameter:

- dataset: brazilian or churn_retail

```
python -m src.classification.online.main_tsunami -dataset brazilian
```
Baseline requere the following parameters:

- dataset: brazilian or churn_retail
- classifier: XGB, LR or RF

```
python -m src.classification.online.main_baseline -dataset brazilian -classifier XGB
```
