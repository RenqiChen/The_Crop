# [CVPR 2024] HiKER-SGG

## 👀Introduction

This repository contains the code for our CVPR 2024 paper `HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation`. [[Paper](https://arxiv.org/abs/2403.12033)] [[Website](https://zhangce01.github.io/HiKER-SGG/)]

![](fig/hikersgg.png)

## 💡Environment

We tested our codebase with PyTorch 1.13.1 and CUDA 11.6. Please install the corresponding versions of PyTorch and CUDA based on your computational resources.

To install the required packages, run:
```bash
pip install -r requirements.txt.
```

This includes jupyter, as you need it to run the notebooks.

### Note
flash-attention need linux kernel higher than 5.5

## ⏳Setup

We use the [COIG-CQIA](https://github.com/paralym/COIG-CQIA) dataset in this work, which consists of multi tasks chinese Instruction Fine-tuning

To use seed data for fine-tuning models, download the seed datasets to the './train/data' folder and revise the file 'dataset_info.json':
file by adding the following annotation to the config file:
```json
  "seed_dataset": {
    "file_name": "seed dataset.json"
  }
```
Here, file_name is the path to the seed dataset. Then, update the training command from:
```bash
 --dataset ruozhiba
```

to
```bash
--dataset ruozhiba,seed_dataset
```

We recommend downloading the pre-trained model weights to the /train/model folder.
```
cd /train
mkdir model
```
then download a pretraining model:
[LLama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
[InternLM2](https://huggingface.co/internlm/internlm2-7b)

## 📦Usage

To train local models using our dataset with LoRA, run:
```
'CUDA_VISIBLE_DEVICES=0 python src/train_bash.py --stage sft --model_name_or_path ./train/model/Meta-Llama-3-8B  --do_train --dataset ruozhiba --finetuning_type lora  --lora_target q_proj,v_proj --output_dir /output --logging_steps 10 --save_steps 100 --num_train_epochs 4 --plot_loss --per_device_train_batch_size=4 --fp16 --template default --preprocessing_num_workers 1'
```
This refined version should help you better understand and utilize the project. If you have any questions, feel free to reach out.
## 📈VG-C Benchmark

In our paper, we introduce a new synthetic VG-C benchmark for SGG, containing 20 challenging image corruptions, including simple transformations and severe weather conditions.

![](fig/corruption.png)

We include the code for generating these 20 corruptions in ``dataloaders/corruptions.py``. To use it, you also need to modify the codes in ``dataloaders/visual_genome.py``, and also enable ``-test_n`` in the evaluation notebook file.

## 🙏Acknowledgements

Our codebase is adapted from [GB-Net](https://github.com/alirezazareian/gbnet) and [EB-Net](https://github.com/zhanwenchen/eoa). We thank the authors for releasing their code!

## 📧Contact

If you have any questions, please  contact at [cezhang@cs.cmu.edu](mailto:cezhang@cs.cmu.edu).

## 📌 BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
@inproceedings{zhang2024hikersgg,
  title={HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation},
  author={Zhang, Ce and Stepputtis, Simon and Campbell, Joseph and Sycara, Katia and Xie, Yaqi},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
This work inspired by [LLama-Facotry](https://github.com/hiyouga/LLaMA-Factory)
