# Gradual Domain Adaptation via Normalizing Flows
Codes for the paper "Gradual Domain Adaptation via Normalizing Flows".

## Requirements
Please check the file named gdacnf_env.yml.  
This file is to create the execution environment.  
>conda env create -n your-env-name -f cnf_env.yml

## Usage
Our experiments consist of two steps as follows.
1. Fit UMAP
2. Train CNF

For starting experiments from 1st step, please download the datasets from the links listed in [the Datasets section](#Datasets).  
After downloading the datasets, run FitUMAP.py to obtain preprocessed datasets.  
Since downloading takes a very long time, it is recommended to use preprocessed datasets in this supplementary material.  
  
The usages of our experimental script are demonstrated in `runExample.sh`.  
We conduct the experiments on our server with Intel Xeon Gold 6354 processors and NVIDIA A100 GPU , and the training of CNF take about 5 hours per dataset.  

Lastly, we can use `MakeFigure.ipynb` to parse the experimental results and obtain the figure shown in our papers.  

<a id="Datasets"></a>
## Datasets
Portraits  
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

SHIFT15M  
https://github.com/st-tech/zozo-shift15m  

RxRx1  
We use WILDS to load pre-processed dataset.  
https://wilds.stanford.edu/datasets/  

Tox21  
We use MoleculeNet to load pre-processed dataset.  
https://moleculenet.org/
