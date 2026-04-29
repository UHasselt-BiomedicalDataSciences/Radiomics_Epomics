# Combining Magnetic Resonance Imaging and Evoked Potentials Enhances Machine Learning Prediction of Multiple Sclerosis Disability Worsening

This folder provides the core code used in the paper:

**Aerts, Werthen-Brabants, Khan et al. (2026). Combining Magnetic Resonance Imaging and Evoked Potentials Enhances Machine Learning Prediction of Multiple Sclerosis Disability Worsening**

## Important Scope Note

This is a **code-only release**.
The underlying patient-level datasets are not included and are not publicly distributed in this repository.

As a result:

- the code is provided for **method transparency**, **reference**, and **verification of implementation details**;
- it can be used for **modification/adaptation** to other datasets;
- it is **not directly reproducible end-to-end** without access to the original non-public data.

## Folder Contents

- `ablation.sh`: shell launcher for the ablation experiment grid.
- `model.py`: model training, hyperparameter search, fold evaluation, and artifact export.
- `utils.py`: data loading, label construction, and preprocessing utilities.
- `figure scripts/`: scripts/notebooks used to generate figures for the manuscript.
- `results_20250204.csv`: results table used by part of the figure generation workflow.

## Intended Use

This release is intended to help readers:

1. inspect how the experiments were implemented;
2. verify analysis logic and modeling choices;
3. reuse or extend the pipeline on their own compatible datasets.

## Running the Code With Your Own Data

If you want to execute the training pipeline, install dependencies from the repository root:

```bash
pip install -r requirements-public.txt
```

Then provide your own processed input datasets expected by `utils.py` (schema-compatible with the original project) and run:

```bash
bash CODE_FOR_PUBLICATION/ablation.sh
```

The code writes experiment artifacts to `results/...` directories.

## Citation

If this code is useful in your work, please cite:

```bibtex
@article{aerts17combining,
  title={Combining Magnetic Resonance Imaging and Evoked Potentials Enhances Machine Learning Prediction of Multiple Sclerosis Disability Worsening},
  author={Aerts, Sofie and Werthen-Brabants, Lorin and Khan, Hamza and Giraldo, Diana L and De Brouwer, Edward and Geys, Lotte and Popescu, Veronica and Sijbers, Jan and Woodruff, Henry and Dhaene, Tom and others},
  journal={Frontiers in Immunology},
  volume={17},
  pages={1625837},
  publisher={Frontiers},
  url={https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2026.1625837/full}
}
```
