import torch
from llm_trainer import TrainerTools
from utils import init_env
from sklearn.utils import shuffle
import json
import itertools
import pickle
import re
import pandas as pd
from constant import system_prompt, reasoning_system_prompt, assistant_name

init_env()
tokenizer = TrainerTools().tokenizer

max_prompt_len = 1024

# 14149024
# sft data: <system>{system_prompt}</s><user>你好</s><assistant>我好</s><user>很好</s><assistant>不好</s>


def _remove_urls(text: str):
    url_pattern = re.compile(r'(?:(?:https?|ftp):\/\/)?[\w\/\-?=%.]+\.[\w\/\-&?=%.]+')
    return url_pattern.sub('', text)
    # urls = extractor.find_urls(text)
    # for url in urls:
    #     text = text.replace(url, "")
    # return text


def _remove_brackets(text: str):
    return (text.replace('[]', '')
     .replace('{}', '')
     .replace('()', '')
     .replace('<>', '')
     .replace('【】', '')
     .replace('《》', '')
     .replace('（）', '')
     .replace('（，）', '')
     # .replace('\"\"', '')
     # .replace("\'\'", '')
     )

def _filter_content(content: str) -> str:
    return _remove_brackets(_remove_urls(content))

def split_data(tag: str):
    data_list_short = []
    data_list_long = []

    print(f'parse sft_data_{tag}')
    with open(f'./data/raw/sft_data_{tag}.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            history = json_['history']
            conversations = []
            content = system_prompt

            for his in history:
                conversations.append({'user': his[0], 'assistant': his[1]})
                content = f'{content}{tokenizer.text_user}{his[0]}{tokenizer.text_end}{tokenizer.text_assistant}{his[1]}{tokenizer.text_end}'

            conversations.append({'user': json_['input'], 'assistant': json_['output']})
            content = f"{content}{tokenizer.text_user}{json_['input']}{tokenizer.text_end}{tokenizer.text_assistant}{json_['output']}{tokenizer.text_end}"

            if len(content) > max_prompt_len:
                data_list_long.append(f'{json.dumps(conversations, ensure_ascii=False)}\n')
            else:
                data_list_short.append(f'{json.dumps(conversations, ensure_ascii=False)}\n')

    # zh long 896730 short 10484891
    # en long 1865052 short 902351
    # total long 2761782 short 11387242
    print(len(data_list_long), len(data_list_short))

    print('dump long')
    with open(f'./data/deepctrl_{tag}_long.jsonl', 'a') as f:
        f.writelines(data_list_long)

    del data_list_long

    print('dump short')
    with open(f"./data/deepctrl_{tag}_short.jsonl", 'a') as f:
        f.writelines(data_list_short)


def merge_long_data():
    print('merge long data')
    # 打开所有文件，并将文件对象存储在一个列表中
    files = ['./data/deepctrl_zh_long.jsonl', './data/deepctrl_en_long.jsonl']
    all_content = []

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                all_content.append(line)

    all_content = shuffle(all_content)
    print('dump')
    with open(f'./data/deepctrl_long.jsonl', 'a') as f:
        f.writelines(all_content)


def encode_long_data():
    print('encode long data')
    # input [{'user':'xxx', 'assistant': xxx}, ...], [...]
    encode_list = []
    suffix = 0
    with open('./data/deepctrl_long.jsonl', 'r') as f:
        for idx, line in enumerate(f):
            conversations = json.loads(line)
            content = ''

            for conversation in conversations:
                item = _filter_content(f"{conversation['user']}{conversation['assistant']}")
                content = f"{content}{item}"

            encode_list.extend(tokenizer.encode_to_token(content, False, False))

            if (idx + 1) % 500000 == 0:
                print(f'dump {suffix}')
                with open(f'./data/deepctrl_long_{suffix}.pkl', 'wb') as f:
                    pickle.dump(encode_list, f)
                encode_list.clear()
                suffix += 1

    print('dump')
    with open('./data/deepctrl_long_final.pkl', 'wb') as f:
        pickle.dump(encode_list, f)


def merge_short_data():
    print('merge short data')
    # 打开所有文件，并将文件对象存储在一个列表中
    files = ['./data/deepctrl_zh_short.jsonl', './data/deepctrl_en_short.jsonl']
    all_content = []

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                all_content.append(line)

    all_content = shuffle(all_content)
    print(f'dump {len(all_content)}')
    with open(f'./data/deepctrl_short.jsonl', 'a') as f:
        f.writelines(all_content)


def encode_short_data():
    print('encode short data')
    def encode_pretrain(pretrain_list):
        print('encode_short_data -> encode_pretrain')
        encode_pretrain_list = []
        suffix = 0
        part_size = len(pretrain_list) // 3

        for idx, line in enumerate(pretrain_list):
            content = ''

            for conversation in line:
                item = _filter_content(f"{conversation['user']}{conversation['assistant']}")
                content = f"{content}{item}"

            content = f'{content}{tokenizer.text_end}'
            encode_pretrain_list.extend(tokenizer.encode_to_token(content, False, False))

            if (idx + 1) % part_size == 0 and suffix < 2:
                print(f'dump {suffix}')
                with open(f'./data/deepctrl_short_{suffix}.pkl', 'wb') as f:
                    pickle.dump(encode_pretrain_list, f)
                encode_pretrain_list.clear()
                suffix += 1

        print('dump')
        with open('./data/deepctrl_short_final.pkl', 'wb') as f:
            pickle.dump(encode_pretrain_list, f)


    def encode_sft(sft_list):
        print('encode_short_data -> encode_sft')
        # input [{'user':'xxx', 'assistant': xxx}, ...], [...]
        # output <system>{system_prompt}</s><user>你好</s><assistant>我好</s><user>很好</s><assistant>不好</s>
        encode_sft_list = []
        for idx, line in enumerate(sft_list):
            content = system_prompt
            for conversation in line:
                user_content = _filter_content(conversation['user'])
                assistant_content = _filter_content(conversation['assistant'])
                content = f"{content}{tokenizer.text_user}{user_content}{tokenizer.text_end}{tokenizer.text_assistant}{assistant_content}{tokenizer.text_end}"

            encode_sft_list.append(tokenizer.encode_to_token(content, False, False))

        del sft_list

        extend_items = [
            f'{system_prompt}<user>你是谁？</s><assistant>我是{assistant_name}，是一款AI大模型</s>',
            f'{system_prompt}<user>你叫什么</s><assistant>我的名字是{assistant_name}</s>',
            f'{system_prompt}<user>你的名字是什么？</s><assistant>我的名字是{assistant_name}</s>',
            f'{system_prompt}<user>你是qwen吗？</s><assistant>不是的，我的名字是{assistant_name}</s>',
            f'{system_prompt}<user>你是deepseek吗？</s><assistant>NO！我的名字是{assistant_name}</s>',
            f'{system_prompt}<user>who are you?</s><assistant>My name is {assistant_name}</s>',
            f'{system_prompt}<user>what\'s your name</s><assistant>My name is {assistant_name}</s>',
        ]

        for item in extend_items:
            encode_sft_list.append(tokenizer.encode_to_token(item, False, False))

        encode_sft_list = shuffle(encode_sft_list)

        print('dump sft')
        with open('./data/sft_deepctrl_short.pkl', 'wb') as f:
            pickle.dump(encode_sft_list, f)


    # 11387242条数据，0.2用于sft，0.8用于预训练
    origin_list = []
    with open('./data/deepctrl_short.jsonl', 'r') as f:
        for idx, line in enumerate(f):
            conversations = json.loads(line)
            origin_list.append(conversations)

    origin_len = len(origin_list)
    pretrain_list = origin_list[:int(origin_len*0.8)]
    sft_list = origin_list[int(origin_len*0.8):]

    encode_pretrain(pretrain_list)
    del pretrain_list
    encode_sft(sft_list)


def encode_dpo_data():
    print('encode dpo data')
    file_path = './data/raw/dpo/dpo.jsonl'
    content_list = []

    chosen_max_len = 0
    reject_max_len = 0

    with open(file_path, 'r') as f:
        for line in f:
            json_ = json.loads(line)

            chosen_list = json_['chosen']
            rejected_list = json_['rejected']

            prompt = ''
            chosen = ''
            rejected = ''

            cur_chosen_len = len(chosen_list)
            cur_reject_len = len(rejected_list)

            if cur_chosen_len > chosen_max_len:
                chosen_max_len = cur_chosen_len

            if cur_reject_len > reject_max_len:
                reject_max_len = cur_reject_len

            for chosen_item in chosen_list:
                if chosen_item['role'] == 'user':
                    prompt = chosen_item['content']
                elif chosen_item['role'] == 'assistant':
                    chosen = chosen_item['content']
            for rejected_item in rejected_list:
                if rejected_item['role'] == 'assistant':
                    rejected = rejected_item['content']

            # <system>{system_prompt}</s><user>你好</s><assistant>
            chosen_content = f'{system_prompt}{tokenizer.text_user}{prompt}{tokenizer.text_end}{tokenizer.text_assistant}{chosen}{tokenizer.text_end}'
            rejected_content = f'{system_prompt}{tokenizer.text_user}{prompt}{tokenizer.text_end}{tokenizer.text_assistant}{rejected}{tokenizer.text_end}'

            chosen = tokenizer.encode_to_token(chosen_content, False, False)
            rejected = tokenizer.encode_to_token(rejected_content, False, False)

            # [{'prompt': [], 'chosen': [], 'rejected': []}]
            content_list.append({'chosen': chosen, 'rejected': rejected})

    files = ['./data/raw/dpo/dpo_zh.json', './data/raw/dpo/dpo_en.json']

    for file in files:
        with open(file, 'r') as f:
            json_ = json.load(f)
            for item in json_:
                prompt = item['conversations'][0]['value']
                chosen = item['chosen']['value']
                rejected = item['rejected']['value']

                chosen_content = f'{system_prompt}{tokenizer.text_user}{prompt}{tokenizer.text_end}{tokenizer.text_assistant}{chosen}{tokenizer.text_end}'
                rejected_content = f'{system_prompt}{tokenizer.text_user}{prompt}{tokenizer.text_end}{tokenizer.text_assistant}{rejected}{tokenizer.text_end}'

                chosen = tokenizer.encode_to_token(chosen_content, False, False)
                rejected = tokenizer.encode_to_token(rejected_content, False, False)

                if len(chosen) > 1024 or len(rejected) > 1024:
                    continue

                content_list.append({'chosen': chosen, 'rejected': rejected})

    content_list = shuffle(content_list)
    with open(f'./data/dpo.pkl', 'wb') as f:
        pickle.dump(content_list, f)

    print(chosen_max_len, reject_max_len)


def encode_reasoning_data():
    print('encode reasoning data')
    # <system>{reasoning_system_prompt}</s><user>你好</s><assistant><reasoning>思考</reasoning><answer>回答</answer></s>
    content_list = []

    with open('./data/raw/r1_mix_1024.jsonl', 'r') as f:
        for line in f:
            conversations = json.loads(line)['conversations']
            conversations_text = reasoning_system_prompt
            for conversation in conversations:
                content = _filter_content(conversation['content'])
                content = content.replace('{{assistant_name}}', assistant_name)
                content = content.replace('<think>', tokenizer.text_reasoning_start).replace('</think>', tokenizer.text_reasoning_end)

                tag = tokenizer.text_user if conversation['role'] == 'user' else tokenizer.text_assistant
                conversations_text = f'{conversations_text}{tag}{content}{tokenizer.text_end}'

            content_list.append(tokenizer.encode_to_token(conversations_text, False, covert_tensor=False))


    with open('./data/raw/alpaca_r1_data_zh-localpost.json', 'r') as f:
        json_ = json.load(f)
        for item in json_:
            instruction = item['instruction']
            content = _filter_content(item['output'])
            content = content.replace('<think>', tokenizer.text_reasoning_start).replace('</think>', tokenizer.text_reasoning_end)

            conversations_text = f'{reasoning_system_prompt}{tokenizer.text_user}{instruction}{tokenizer.text_end}{tokenizer.text_assistant}{content}{tokenizer.text_end}'
            encoded = tokenizer.encode_to_token(conversations_text, False, covert_tensor=False)
            if len(encoded) <= 1024:
                content_list.append(encoded)

    with open(f'./data/r1_mix_1024.pkl', 'wb') as f:
        pickle.dump(content_list, f)


def encode_grpo_data():
    print('encode grpo data')
    # <system>{reasoning_system_prompt}</s><user>你好</s><assistant><reasoning>思考</reasoning><answer>回答</answer></s>
    file_names = ['train-00000-of-00001.parquet', 'test-00000-of-00001.parquet']
    qas = []
    for file_name in file_names:
        df = pd.read_parquet(f"./data/raw/gsm8k_chinese/{file_name}", engine="pyarrow")
        # 'question_zh-cn', 'answer_only'
        for q, a in zip(df['question_zh-cn'].values, df['answer_only'].values):
            # <system>{reasoning_system_prompt}</s><user>你好</s><assistant><reasoning>
            q = f'{reasoning_system_prompt}<user>{q}</s><assistant>'
            a = str(a)
            item = {
                'prompt': tokenizer.encode_to_token(q, False, covert_tensor=False),
                'answer': tokenizer.encode_to_token(a, False, covert_tensor=False),
            }

            qas.append(item)

    with open(f'./data/grpo.pkl', 'wb') as f:
        pickle.dump(qas, f)


def test_pretrain():
    from llm_trainer.dataset import TextDataset
    pretrain_data_files = [
        # './data/deepctrl_long_0.pkl',
        # './data/deepctrl_long_1.pkl',
        # './data/deepctrl_long_2.pkl',
        # './data/deepctrl_long_3.pkl',
        # './data/deepctrl_long_4.pkl',
        # './data/deepctrl_long_final.pkl',
        './data/deepctrl_short.pkl',
    ]

    for file in pretrain_data_files:
        dataset = TextDataset(file, 1024, 1024)
        print(f'{file}, {len(dataset)}, {tokenizer.decode_to_text(dataset.__getitem__(0))}')


def test_sft():
    from llm_trainer.dataset import LineByLineTextDataset
    sft_data_file = './data/sft_deepctrl_short.pkl'

    dataset = LineByLineTextDataset(sft_data_file, 1024)
    print(f'{len(dataset)}, {tokenizer.decode_to_text(dataset.__getitem__(0))}')


def test_dpo():
    from llm_trainer.dataset import DPODataset
    dpo_data_file = './data/dpo.pkl'

    dataset = DPODataset(dpo_data_file, 1024)
    item = dataset.__getitem__(0)
    chosen = tokenizer.decode_to_text(torch.tensor(item['chosen']))
    reject = tokenizer.decode_to_text(torch.tensor(item['rejected']))

    print(f'{len(dataset)}, chosen={chosen}, reject={reject}')


def test_reasoning():
    from llm_trainer.dataset import LineByLineTextDataset
    reasoning_data_file = './data/r1_mix_1024.pkl'

    dataset = LineByLineTextDataset(reasoning_data_file, 1024)
    print(f'{len(dataset)}, {tokenizer.decode_to_text(dataset.__getitem__(0))}')


def test_grpo():
    from llm_trainer.dataset import GRPORolloutDataset

    dataset = GRPORolloutDataset('./data/grpo.pkl')
    item = dataset.__getitem__(0)
    prompt = tokenizer.decode_to_text(item['prompt'])
    answer = tokenizer.decode_to_text(item['answer'])


    print(f'{len(dataset)}, prompt={prompt}, answer={answer}')


if __name__ == '__main__':
    print(tokenizer.encode_to_token('<system>1</s><user>1</s><assistant><reasoning>1</reasoning><answer>1</answer>'))
    # split_data('zh')
    # split_data('en')
    # merge_long_data()
    # merge_short_data()
    # encode_long_data()
    # encode_short_data()
    encode_dpo_data()
    # encode_reasoning_data()
    # encode_grpo_data()

    # test_pretrain()
    # test_sft()
    # test_dpo()
    # test_reasoning()
    # test_grpo()
