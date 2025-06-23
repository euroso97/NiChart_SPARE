# NiChart_SPARE

Implementation of SPARE scores calculation from Brain ROI Volumes ([NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE)) and White-Matter-Lesion volumes ([NiChart_DLWMLS](https://github.com/CBICA/NiChart_DLWMLS)) as main features.

### Supported SPARE scores (as of June 2025):
- SPARE-BA: Brain Age
- SPARE-AD: Alzheimer's 
- SPARE-HT: Hypertension
- SPARE-HL: Hyperlipidemia
- SPARE-T2B: Diabetes (Type 2)
- SPARE-SM: Smoking
- SPARE-OB: Obesity

## Installation

##### Using PyPi
```bash
pip install NiChart_SPARE
```

##### From GitHub
```bash
git clone https://github.com/CBICA/NiChart_SPARE.git
cd NiChart_SPARE
pip install -e .
```

### Example Usage
##### Training
```bash
NiChart_SPARE   -a      trainer                 \
                -t      AD                      \
                -i      /path/to/input_file.csv \
                -mo     /path/to/spare_ad_model.pkl.gz
                -v      True
```
##### Inference
```bash
NiChart_SPARE   -a      inference                       \
                -t      AD                              \
                -i      /path/to/test_file.csv          \
                -m      spare_ad_model.pkl.gz           \
                -v      False                           \
                -o      /path/to/output_spare_score.csv
                
```
## Documentation

Coming Soon (Wiki-page)

## Publications

- SPARE-BA

  Habes, M. et al. Advanced brain aging: relationship with epidemiologic and genetic risk factors, and overlap with Alzheimer disease atrophy patterns. Transl Psychiatry 6, e775, [doi:10.1038/tp.2016.39](https://doi.org/10.1038/tp.2016.39) (2016).

- SPARE-AD

  Davatzikos, C., Xu, F., An, Y., Fan, Y. & Resnick, S. M. Longitudinal progression of Alzheimer's-like patterns of atrophy in normal older adults: the SPARE-AD index. Brain 132, 2026-2035, [doi:10.1093/brain/awp091](https://doi.org/10.1093/brain/awp091) (2009).

- diSPARE-AD

  Hwang, G. et al. Disentangling Alzheimer's disease neurodegeneration from typical brain ageing using machine learning. Brain Commun 4, fcac117, [doi:10.1093/braincomms/fcac117](https://doi.org/10.1093/braincomms/fcac117) (2022).

- SPARE-CVMs (HT, HL, T2B, SM, OB)

  Govindarajan, S.T., Mamourian, E., Erus, G. et al. Machine learning reveals distinct neuroanatomical signatures of cardiovascular and metabolic diseases in cognitively unimpaired individuals. Nat Commun 16, 2724, [doi:10.1038/s41467-025-57867-7](https://doi.org/10.1038/s41467-025-57867-7) (2025). 


  # Notes

  - data_prep.py : subsets data for training/cross-validation and also perform additional processes including standardscaling, adjustment of Age/Sex/ICV effects 

  - pipelines/spare_(BIOMARKER).py : (Biomarker) specific training pipelines

  - __main__.py : entry point for CLI, handle input arguments and calling of specific spare training pipeline or inferencing code