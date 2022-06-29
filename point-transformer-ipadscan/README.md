# Point Transformer
This repository reproduces [Point Transformer](https://arxiv.org/abs/2012.09164). \
The codebase is provided by the first author of [Point Transformer](https://arxiv.org/abs/2012.09164).

## Dependencies
- Ubuntu: 18.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1
- Hardware: 4GPUs (TITAN RTX) to reproduce [Point Transformer](https://arxiv.org/abs/2012.09164) 
- To create conda environment, command as follows:

  ```
  sh env_setup.sh pt
  ```

## Usage

> Command reference. 
> If u want to run these codes on ur machine, some settings need to be modified.

- **train**: `sh tool/train.sh ipad_scaned baseline5`

- **test**: `sh tool/test_ipadscan.sh exp/ipad_scaned/baseline5/`

- **visualization**: `python tool/test_generate_vis.py`

- **calculation index**: `python tool/test_calc_index.py`


## References

If you use this code, please cite [Point Transformer](https://arxiv.org/abs/2012.09164):
```
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```

## Acknowledgement
The code is from the first author of [Point Transformer](https://arxiv.org/abs/2012.09164).
