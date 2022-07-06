# BEHAVE
Repo for BEHAVE: Dataset and Method for Tracking Human Object Interactions, CVPR'22

Link to paper: https://arxiv.org/pdf/2204.06950.pdf

## Prerequisites
- I've added my environment files to requirements.txt. You can directly install these packages. Some of them are not necessary but good to have for debugging, visualizations etc. Core packages are listed below.
- Cuda 10.2
- Python 3.7
- Pytorch 1.7.1
- pytorch3d 0.2.0
- MPI mesh library (https://github.com/MPI-IS/mesh) This is good to have. you can also replace it with trimesh.
- Trimesh
- SMPL pytorch from https://github.com/gulvarol/smplpytorch. I have included these files (with required modifications) in this repo.
- Download SMPL from https://smpl.is.tue.mpg.de/
- Use the script utils/voxelize_ho.py to voxelize the human and object point cloud from BEHAVE dataset. This is the input to the network.
- Use the scipt utils/compute_df_ho.py to sample query points and compute distance and correspondence fields. This is the supervision to the network.
- Prepare diffused SMPL from LoopReg, NeurIPS'20, with the script utils/spread_SMPL_function.py
- Make data split for training and testing using the script utils/make_data_split.py
- Download assets: Coming soon.

## Download pre-trained models
Coming soon.

## Test BEHAVE model
```python train.py -mode val -exp_id 01 -ext 01 -suffix 01 -save_name val -split_file assets/data_split_01.pkl -batch_size 12```
## Train BEHAVE model
```python train.py -exp_id 01 -ext 01 -suffix 01 -split_file assets/data_split_01.pkl -batch_size 32 -epochs 150```

## Cite us:
```
@inproceedings{bhatnagar22behave,
    title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
    author = {Bhatnagar, Bharat Lal and  Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022},
}
```

## LICENCE
Copyright (c) 2020 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the "BEHAVE: Dataset and Method for Tracking Human Object Interactions" paper in documents and papers that report on research using this Software.

