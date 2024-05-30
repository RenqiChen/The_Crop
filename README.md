# [Submitted to NIPS2024 Dataset & Benchmark Track] The Seed.

## 👀Introduction

This repository contains the code and data for our NIPS 2024 paper `The Seed`.
## 📚 Dataset
In our work, we propose a instruction fine-tuning dataset for crop cultivation.
# Composition of the Single-round Dialogue Dataset
Please note that despite our data-cleaning efforts, the final QA pairs inevitably contain a small amount of data (<0.5%) from other grains like wheat. Because we think this data does not dominantly influence the fine-tuning results, it is included in the rice and maize QA pairs, and we have listed it separately to avoid any misleading counts.

| Cereal | Type | Task                        | English QA | Chinese QA | Total  |
|--------|------|-----------------------------|------------|------------|--------|
| Rice   | Closed QA | In-domain knowledge QA  | 42951      | 83396      | 126347 |
|        | Open QA   | In-domain knowledge QA  | 2430       | 2037       |        |
|        |           | Event extraction        | 1891       | 1030       |        |
|        |           | Named Entity Recognition| 2003       | 1604       |        |
|        |           | Summary                 | 1586       | 1628       | 14209  |
| Corn   | Closed QA | In-domain knowledge QA  | 25259      | 27667      | 52926  |
|        | Open QA   | In-domain knowledge QA  | 3202       | 3047       |        |
|        |           | Event extraction        | 2245       | 1322       |        |
|        |           | Named Entity Recognition| 2008       | 1316       |        |
|        |           | Summary                 | 1559       | 1857       | 16556  |
| Others*| ---      | ---                      | ---        | ---        | <1000  |
| Overall| ---      | ---                      | 85134      | 124904     | 210038 |

*The 'Others' category includes data from other grains like wheat, which is less than 1000 QA pairs in total.


## 💡Environment

We tested our codebase with PyTorch 1.13.1 and CUDA 11.6. Please install the corresponding versions of PyTorch and CUDA based on your computational resources.

To install the required packages, run:
```bash
pip install -r requirements.txt.
```

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
[LLama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) ,
[InternLM2-7B](https://huggingface.co/internlm/internlm2-7b) ,
[Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B) .

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
