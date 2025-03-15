import random
import jsonlines


first_dataset = []
sec_dataset = []
cnt = 0

with jsonlines.open('./DISC-Law-SFT-Pair.jsonl') as reader:
    for each in reader:
        if "jud_read_compre" in each["id"]:
            real_data = {}
            real_data["instruction"] = "阅读下述案件背景，然后回答问题"
            real_data["input"] = f"{each['input']}"
            real_data["output"] = f"{each['output']}"
            first_dataset.append(real_data)
            cnt += 1

with jsonlines.open("./big_train_data.json") as reader:
    for each in reader:
        for item in each["data"]:
            for each_case in item["paragraphs"]:
                for each_question in each_case["qas"]:
                    if len(each_question['answers']) == 0:
                        continue
                    real_data = {}
                    real_data["instruction"] = "阅读下述案件背景，然后回答问题"
                    real_data["input"] = (f"{each_case['context']}\n"
                                          f"请回答下述问题：{each_question['question']} ")
                    real_data["output"] = f"{each_question['answers'][0]['text']}"
                    sec_dataset.append(real_data)

dataset = first_dataset + sec_dataset

with jsonlines.open("dataset/dataset.jsonl", "w") as writer:
    for each in dataset:
        writer.write(each)


dataset = []

with jsonlines.open("dataset/dataset.jsonl") as reader:
    for each in reader:
        dataset.append(each)

print(len(dataset))

length = len(dataset)
random.seed(42)
random.shuffle(dataset)

train = dataset[:int(length*0.8)]
test = dataset[int(length*0.8):]

with jsonlines.open("./train.jsonl", "w") as writer:
    for each_line in train:
        writer.write(each_line)

with jsonlines.open("dataset/test.jsonl", "w") as writer:
    for each_line in test:
        writer.write(each_line)
