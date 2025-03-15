from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import Dataset
import torch
import jsonlines

max_seq_length = 2048
dtype = None
load_in_4bit = True

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def train():
    dataset = []

    with jsonlines.open("./dataset/train.jsonl") as reader:
        for each_line in reader:
            dataset.append(each_line)

    model, tokenizer = FastLanguageModel.from_pretrained(
        # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
        # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
        model_name = "unsloth/Qwen2.5-1.5B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    dataset = Dataset.from_list(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    model.save_pretrained("model_parameters")
    tokenizer.save_pretrained("model_parameters")


def test():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./model_parameters",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    # alpaca_prompt = You MUST copy from above!

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "阅读下述案件背景，然后回答问题",  # instruction
                "聚众斗殴罪.经审理查明：一、聚众斗殴2016年2月20日下午，被告人孙1某因与刘某发生口角后心怀不满，便联系被告人宋某让其喊人帮忙教训刘某。被告人宋某电话联系郭某（另案处理）让其接鞠某（绰号“校长”，已上网追逃）。郭某随后纠集郑某接到鞠某；被告人宋某接到李某（另案处理）一同来到南通市通州区川姜镇磨框大桥与被告人孙1某及郭某等人汇合。郭某从被告人宋某车内拿了一把砍刀后驾车带着郑某和鞠某，被告人宋某驾车带着李某，被告人孙1某独自驾车来到南通市通州区川姜镇姜灶煤气站对面××轮胎店门口刘某所在地。被告人孙1某手持棒球棍殴打刘某腿部、头部；郭某持砍刀冲到刘某右后侧并殴打刘某。刘某一方的朱某上前劝阻，被郭某用砍刀砍伤左侧脸部和左手背。之后，郭某继续持砍刀并殴打刘某；被告人孙1某持棒球棍、李某持棍子共同殴打刘某。郭某、鞠某追打朱某，鞠某使用拖把击打朱某并将其踹倒在超市门口的河边；郭某持砍刀并殴打朱某的背部、腰部；被告人孙1某手持关某刀指着朱某的脸对其进行辱骂。随后，众人离开案发现场。后经鉴定，刘某损伤程度为轻伤一级，朱某损伤程度为轻微伤。事发后，被告人宋某、孙1某赔偿了伤者刘某等人部分经济损失。. this is the question: 刘某和朱某的受伤情况？",
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    output = model.generate(**inputs, max_new_tokens=128)
    tokenizer.batch_decode(output)