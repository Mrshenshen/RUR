import os
import json
import argparse
from tqdm import tqdm
import re
parser = argparse.ArgumentParser()

parser.add_argument("--gt_files", type=str, default="/path/to/gt pope coco") 
parser.add_argument("--gen_files", type=str, default="/your/path/to/pope results/.json")

args = parser.parse_args()

# open ground truth answers
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
yes_label = 0
total_questions = len(gen_files)  #len(gen_files)
wrong_number=""
yes_answers = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    if idx>=total_questions:
        break
    gt_answer = line["label"]
    # assert idx == gen_files[index]["question_id"]
    gen_answer = gen_files[index]["answer"]
    # score = gen_files[index]["clip_score"]
    g = gen_answer
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    t=gen_answer.replace('.', '')
    t=t.replace(',', '')
    t=t.split(' ')
    gen_answer=t[0]
    
    # strip
    gt_answer = gt_answer.strip()

    
    if gt_answer == 'yes':
        yes_label += 1
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        elif 'no' in gen_answer:
            false_neg += 1
            # if "Without logic" in g:
                # print("false_neg:",idx," score:",score)
            wrong_number+=str(idx)+", "
        else:
            unknown +=1

    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        elif 'yes' in gen_answer:
            yes_answers += 1
            false_pos += 1
            # print("false_pos:",idx)
            wrong_number+=str(idx)+", "
        else:
            unknown+=1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions

# report results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print(f'unknow: {unknown_prop}')
print(f'yes in label: {yes_label/total_questions}')
print(wrong_number)