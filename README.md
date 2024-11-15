Codes for Generative Embedding (using DCM-based connectivity features) with Clustering Solution (GECS). In this repository, we specifically use P-DCM [1] which is a state-of-the-art DCM framework.

The overall generative embedding process comprises of:

<strong>Step 1.</strong> Computation of connectivity features for task-fMRI data of several subjects in an iterative manner using the DCM framework(s).

P-DCM was initially developed on Matlab and the link to the corresponding codebase is: https://github.com/BRAIN-TO/PDCM

For S-DCM [2], one may refer to the standard SPM package which involves the code for running S-DCM on task fMRI data.

Alternatively, one may also use the python implementation of both P-DCM and S-DCM which are available in the src/utilities folder in this repository.

<strong>Step 2.</strong> Obtain the clusters based on the estimated connectivity features.

Once, the connectivity estimates are obtained it is recommended to store them in a pandas dataframe as a '.pt' file with subjects in rows and connectivity between different regions in the columns.

This dataframe is to be used in the src/EC_cluster.py script to obtain the clusters by utilizing k-means and leave-one-out-cross validation strategy.

The Accuracy and Cluster Membership Index (CMI) scores are also computed simultaneously.

References:

```
[1] Havlicek, M., Roebroeck, A., Friston, K., Gardumi, A., Ivanov, D., & Uludag, K. (2015). Physiologically informed dynamic causal modeling of fMRI data. Neuroimage, 122, 355-372.

[2] Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. Neuroimage, 19(4), 1273-1302.
```
