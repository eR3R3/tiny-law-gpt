import unsloth
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import Dataset
from tqdm import tqdm
import torch
import jsonlines
from rouge import Rouge
import jieba
import numpy as np

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
        model_name = "unsloth/Qwen2.5-32B",
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


def save_prediction_with_finetune():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model_parameters",  
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model) 

    test_dataset = []
    predictions = []

    with jsonlines.open('dataset/test.jsonl') as reader: 
        for each_line in reader:
            test_dataset.append(each_line)

    for each in tqdm(test_dataset[:100]):
        inputs = tokenizer([alpaca_prompt.format(each["instruction"], each["input"], "")], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=128)
        output = tokenizer.batch_decode(output)[0]
        start = output.find("### Response:")+len("### Response:")
        end = output.find('<|endoftext|>')
        if end == -1: 
            end = len(output)
        output = output[start: end].strip()
        predictions.append(output)

    with jsonlines.open('prediction.jsonl', 'w') as writer:
        for each in predictions: 
            writer.write(each)

            
def save_prediction_without_finetune():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-32B",  
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model) 

    test_dataset = []
    predictions = []

    with jsonlines.open('dataset/test.jsonl') as reader: 
        for each_line in reader:
            test_dataset.append(each_line)

    for each in tqdm(test_dataset[:100]):
        inputs = tokenizer([alpaca_prompt.format(each["instruction"], each["input"], "")], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=128)
        output = tokenizer.batch_decode(output)[0]
        start = output.find("### Response:")+len("### Response:")
        end = output.find('<|endoftext|>')
        if start == -1:
            print("生成失败。" + output)
            continue
        if end == -1: 
            end = len(output)
        output = output[start: end].strip()
        predictions.append(output)
        
    with jsonlines.open('prediction_no_finetune.jsonl', 'w') as writer:
        for each in predictions: 
            writer.write(each)


def calculate_score():
    real = []
    prediction = []
    prediction_without_finetune = []

    f1_score0 = []
    f1_score1 = []

    with jsonlines.open('dataset/test.jsonl') as reader:
        for each in reader:
            real.append(" ".join(jieba.lcut(each["output"])))
        
    
    with jsonlines.open('prediction.jsonl', 'r') as reader:
        for each in reader: 
            prediction.append(" ".join(jieba.lcut(each)))

    with jsonlines.open('prediction_no_finetune.jsonl', 'r') as reader: 
        for each in reader:
            prediction_without_finetune.append(" ".join(jieba.lcut(each)))

    rouge = Rouge()   
    for i in range(100):
        score0 = rouge.get_scores(real[i], prediction[i])
        score0_f1_score = score0[0]["rouge-l"]["f"]
        f1_score0.append(score0_f1_score)
        
        score1 = rouge.get_scores(real[i], prediction_without_finetune[i])
        score1_f1_score = score1[0]["rouge-l"]["f"]
        f1_score1.append(score1_f1_score)

    mean0 = np.mean(f1_score0)
    mean1 = np.mean(f1_score1)
    
    print(f"prediction with finetune: {f1_score0} \n, prediction without finetune{f1_score1} \n")
    print(f"mean0: {mean0}, mean1: {mean1}")


def dummy_calculate_score():
    a = ["something", "ggbond"]
    b = ["ggbond", "ccbond"]
    rouge = Rouge()
    
#train()
#test()
#save_prediction_with_finetune()
#save_prediction_without_finetune()

calculate_score()
























