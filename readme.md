# SCSMDA
**SCSMDA: Predicting microbe-drug associations with structure-enhanced contrastive learning and self-paced negative sampling strategy.**

### Dataset ###
  * MDAD: MDAD has 1373 drugs and 173 microbes with 2470 observed drug-microbe pairs
  * aBiofilm: aBiofilm has 1720 drugs and 140 microbes with 2884 observed drug-microbe pairs
  * DrugVirus: DrugVirus has 175 drugs and 95 microbes with 933 observed drug-microbe pairs

### Data description ###
* adj: microbe and drug interactions
* drugs: drug name and corresponding id
* microbes/viruses: microbes/viruses name and corresponding id
* drugfeatures: pre-processed feature matrix for drugs.
* microbefeatures: pre-processed feature matrix for microbes.
* drugsimilarity: integrated drug similarity matrix.
* microbesimilarity: integrated microbe similarity matrix.

### Run Step ###
  Run train.py to train the model and obtain the predicted scores for microbe-drug associations.


### Requirements ###
  - python==3.9.5
  - pytorch==1.11.0 
  - pyg==2.0.4
  - scikit-learn==1.1.1
  - numpy==1.22.3


### Citation ###
Please kindly cite the paper if you use refers to the paper, code or datasets.


