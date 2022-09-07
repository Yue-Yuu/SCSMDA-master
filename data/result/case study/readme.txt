
'drug2microbe_A_top_20.txt' contains the top 20 Microbes corresponding to each drug in dataset A predicted by using the trained model. 
In this file, each row has four elements:[drug_name_d, microbe_name_m, rank, score]; 
  'score' is the predicted probability for drug d and microbe m to have relation
  'rank' is the  ranking of the scores of drug d and all remain microbes after removing the positive samples from the training set


'microbe2drug_A_top_20.txt' contains similar data as 'drug2microbe_A_top_20.txt',but with different order.
In microbe2drug_*, the four elements of each row are arranged in the following order:
  [microbe_name_m, drug_name_d, rank, score], represents the predicted score and ranking of each drug d for microbe m