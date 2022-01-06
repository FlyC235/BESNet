# Boundary Enhancing Semantic Context Network for Parsing High-resolution Remote Sensing Images
- Updating......

The code is based on FCN_8s

# Datasets
- ISPRS [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/) dataset
- ISPRS [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) dataset
- The original dataset can be requested for the download from [here](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
- You should cut the training images as well as corresponding labels into patches with an overlap of 171 pixels.

# Requiements
- Python == 3.7.10
- PyTorch == 1.8.1
- CUDA ==10.1

# Train
For Vaihingen, run:
```
sh train_vaihingen_fcn.sh
```
For Potsdam, run:
```
sh train_potsdam_fcn.sh
```
The results will be saved in the `./snapshots/` folder.

# Test
For test, run:
```
sh Eval1.sh
```
For different datasets, please manully change `dataset` in `Eval1.sh`.