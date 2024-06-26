Drug-Induced Gene Expression Ranking Analysis (DIGERA)
=======================
![DIGERA](https://github.com/hclim0213/DIGERA/assets/48709737/1b185498-8d61-4c01-b526-a828197493bd)

What is DIGERA?
----------------

**Drug-Induced Gene Expression Ranking Analysis (DIGERA)** is a Lasso-based ensemble framework utilizing LINCS L1000 data to predict rankings of drug-induced gene expression by combining Graph Convolutional Network, Graph-based Transformer, Transformer Encoder, LSTM, and Random Forest Regressors.

The Library of Integrated Network-Based Cellular Signatures ([LINCS](https://lincsportal.ccs.miami.edu/datasets/)) provide a comprehensive repository of cellular response patterns to various perturbagens and introduce the [L1000](https://www.sciencedirect.com/science/article/pii/S0092867417313090) assay, which measures the expression of 978 landmark genes for high-throughput profiling.

We used [the Bayesian-based curated level 5 dataset](https://academic.oup.com/bioinformatics/article/36/9/2787/5719019) containing differential gene expression signatures and selected profiles from the eight largest cell lines (A375, A549, HA1E, HEPG2, HT29, MCF7, PC3, and VCAP) at concentrations of 5 and 10 μM, with exposure times of 6 and 24 hours.

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

To check out the latest DIGERA sources:

   `$ git clone https://github.com/hclim0213/DIGERA DIGERA`

How to Cite
----------
Hyein Cho, Kyoung Tai No, and Hocheol Lim. "Development of Drug-Induced Gene Expression Ranking Analysis (DIGERA) and Its Application to Virtual Screening for Poly (ADP-Ribose) Polymerase 1 Inhibitors"
