# llm-finetune
开源大模型微调

使用[qwen2](https://qwen.readthedocs.io/zh-cn/latest/training/SFT/example.html)官方教程微调

## 系统环境
### 硬件
- CPU：Intel(R) Xeon(R) Platinum 8480+
- 内存：2T
- NVIDIA H800(80G) * 8，
### 软件
- python：3.10.14
- CUDA Version：12.4
- transformers: 4.42.4
- torch: 2.3.1
- peft: 0.11.1
- deepspeed: 0.14.4
- accelerate: 0.33.0
- optimum: 1.21.2



## 数据集

Ape210K 是一个新的大规模和模板丰富的数学单词问题数据集，包含 210K 个中国小学水平的数学问题。每个问题都包含最佳答案和得出答案所需的方程式。

通过[data_process_qwen2](./notebook/data_process_qwen2.ipynb)将Ape210K数据转换成qwen2微调所需格式。



## 训练
命令：`bash finetune.sh -m /mnt/models/Qwen2-72B-Instruct -d /root/projects/train/Qwen2/data/APE2k/train.jsonl --deepspeed /root/projects/train/Qwen2/examples/sft/ds_config_zero3.json --use_lora True`

参照官方提示：
```
cd examples/sft
bash finetune.sh -m <model_path> -d <data_path> --deepspeed <config_path> [--use_lora True] [--q_lora True]
```
为您的模型指定 <model_path> ，为您的数据指定 <data_path> ，并为您的Deepspeed配置指定 <config_path> 。如果您使用LoRA或Q-LoRA，只需根据您的需求添加 --use_lora True 或 --q_lora True 。这是开始微调的最简单方式。如果您想更改更多超参数，您可以深入脚本并修改这些参数。

### shell脚本
在展示Python代码之前，我们先对包含命令的Shell脚本做一个简单的介绍。我们在Shell脚本中提供了一些指南，并且此处将以 finetune.sh 这个脚本为例进行解释说明。

要为分布式训练（或单GPU训练）设置环境变量，请指定以下变量： GPUS_PER_NODE 、 NNODES、NODE_RANK 、 MASTER_ADDR 和 MASTER_PORT 。不必过于担心这些变量，因为我们为您提供了默认设置。在命令行中，您可以通过传入参数 -m 和 -d 来分别指定模型路径和数据路径。您还可以通过传入参数 --deepspeed 来指定Deepspeed配置文件。我们为您提供针对ZeRO2和ZeRO3的两种配置文件，您可以根据需求选择其中之一。在大多数情况下，我们建议在多GPU训练中使用ZeRO3，但针对Q-LoRA，我们推荐使用ZeRO2。

有一系列需要调节的超参数。您可以向程序传递 --bf16 或 --fp16 参数来指定混合精度训练所采用的精度级别。此外，还有其他一些重要的超参数如下：

- `--output_dir`: the path of your output models or adapters.
- `--num_train_epochs`: the number of training epochs.
- `--gradient_accumulation_steps`: the number of gradient accumulation steps.
- `--per_device_train_batch_size`: the batch size per GPU for training, and the total batch size is equalt to `per_device_train_batch_size * number_of_gpus * gradient_accumulation_steps`
- `--learning_rate`: the learning rate.
- `--warmup_steps`: the number of warmup steps.
- `--lr_scheduler_type`: the type of learning rate scheduler.
- `--weight_decay`: the value of weight decay.
- `--adam_beta2`: the value of in Adam.
- `--model_max_length`: the maximum sequence length.
- `--use_lora`: whether to use LoRA. Adding --q_lora can enable Q-LoRA.
- `--gradient_checkpointing`: whether to use gradient checkpointing.



## 预测

使用[predict](./predict.py)脚本加载训练后的模型进行测试