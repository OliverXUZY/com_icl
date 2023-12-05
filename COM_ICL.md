# Language Model Evaluation Harness

## Basic Usage
### run tasks

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=gpt2 \
    --tasks pr \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python submit_euler.py --run_bash run_cuda0.sh

python submit_euler.py --run_bash run_cuda1.sh --gpu_num 2

python submit_euler.py --run_bash run_cuda0.sh --gpu_num 4
```

### aggregate result
get accuracy from write_out.json, modify the output_dir
```bash
python tools/acc.py
```

get wer from write_out.json, modify the output_dir
```bash
python tools/wer.py
```

## Modified from 

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
