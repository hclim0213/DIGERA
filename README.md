Drug-Induced Gene Expression Ranking Analysis (DIGERA)
=======================

What is DIGERA?
----------------

**Drug-Induced Gene Expression Ranking Analysis (DIGERA)** is a Lasso-based ensemble framework utilizing LINCS L1000 data to predict rankings of drug-induced gene expression by combining Graph Convolutional Network (GCN), Graph-based Transformer, Transformer Encoder, LSTM, and Random Forest Regressors.
The Library of Integrated Network-Based Cellular Signatures (LINCS) provide a comprehensive repository of cellular response patterns to various perturbagens and introduce the L1000 assay, which measures the expression of 978 landmark genes for high-throughput profiling (https://lincsportal.ccs.miami.edu/datasets/).
We used the Bayesian-based curated level 5 dataset of Qiu et al. (https://academic.oup.com/bioinformatics/article/36/9/2787/5719019) containing differential gene expression signatures and selected profiles from the eight largest cell lines (A375, A549, HA1E, HEPG2, HT29, MCF7, PC3, and VCAP) at concentrations of 5 and 10 μM, with exposure times of 6 and 24 hours.

Contact Person
--------------
* Dr. Hocheol Lim (ihc0213@yonsei.ac.kr)
* Ms. Hyein Cho (hicho@bmdrc.org)
* Prof. Kyoung Tai No (ktno@yonsei.ac.kr)

Acknowledgments
---------------
This research was supported by Quantum Advantage challenge research based on 
Quantum Computing through the National Research Foundation of Korea (NRF) 
funded by the Ministry of Science and ICT (RS-2023-00257288).

Prerequisites
-------------
* LINUX/UNIX Cluster Machines
* Python

How to Get
----------

DIGERA program is available through
the repositories hosted on
github ( https://github.com/hclim0213/DIGERA ).

To check out the latest FMO/VQE sources:

   `$ git clone https://github.com/hclim0213/DIGERA DIGERA`

How to Cite
----------
