import base64
import requests
from PIL import Image
from io import BytesIO

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from openai import OpenAI
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json

from lib.clip_utils import CLIPModel
import time

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}

client = OpenAI(
      api_key="sk-xxxxxxxxxxxxxxxx"
)
NUM_SECONDS_TO_SLEEP = 0.5

SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."


GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinationsshould be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''

PROMPT_HISTORY_TEMPLATE_INSTRUCTBLIP= """
You are an image analyst. You need to observe the image carefully and answer question.
<ImageHere>{question} 
"""
PROMPT_HISTORY_TEMPLATE_MINIGPT4 = """
###Human: <Img><ImageHere></Img> {question} ###Assistant:
"""


PROMPT_HISTORY_TEMPLATE_LLAVA = """
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, 
and polite answers to the human's questions and briefly describes the characteristics of the entity.
USER:
<ImageHere>
{question}
ASSISTANT:
"""

PROMPT_TEMPLATE_TARGET = """"
You are given a sentence, extract the entities within the sentence for me. 
[Task]
Your task is to extract the common objects and summarize them as general categories without repetition, merging essentially similar objects.
Avoid extracting abstract or non-specific entities. 
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None".
DO NOT RESPOND WITH ANYTHING ELSE.

Here are examples:
[Sentence]:
The image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.
[Response]:
man.motorcycle
[Sentence]:
There are a few people around, including one person standing close to the motorcyclist and another person further away.
[Response]:
person.motorcyclist
[Sentence]:
Is there a car in this image? No, there is no car in the image. There are a woman in this image.
[Response]:
car.woman
[Sentence]:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed.
[Response]:
dog.cat.bed

Now complete the following: 
[Sentence]:
{sent}
[Response]:
"""

PROMPT_TEMPLATE_FACTOR_ATTRIBUTE = """
You will receive a piece of text, extract the sentences that describes the given object.
[Task]
Your task is to accurately identify and extract every attribute associated with the given object in the provided text. 
Each claim should be an affirmative sentence(without 'no' or 'not'), concise (less than 15 words) and self-contained, corresponding to only one attribute. 
You MUST only respond in the format as required.Each line should contain the original claim and the modified question based on the original claim. 
DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED.
TRANSFORM NEGATIVE SENTENCES INTO AFFIRMATIVE SENTENCES.

[Response Format]
original claim&modified question

Here are examples:
[Text]:
Yes, there is a person in the image. The person is a woman. The woman is walking with a bag and holding an umbrella.
[Entity]:
woman
[Response]:
The woman is walking with a bag.&Is the woman walking with a bag?
The woman is holding an umbrella.&Is the woman holding an umbrella?
[Text]:
No, there is no bird in the image.
[Entity]:
bird
[Response]:
There is a bird in the image.&Is there a bird in the image?
[Text]:
The truck in the picture is a red and black pickup truck, which is parked in a driveway. It is a small truck, likely a compact or mid-size model, and appears to be in good condition.
[Entity]:
truck
[Response]:
The truck is red and black.&Is the truck red and black?
The truck is a pickup truck.&Is the truck a pickup truck?
The truck is parked in a driveway.&Is the truck parked in a driveway?
The truck is a small truck.&Is the truck a small truck?
The truck appears to be in good condition.&Is the truck appears to be in good condition?
[Text]:
No, there is no baseball in the image.
[Entity]:
baseball
[Response]:
There is a baseball in this image.&Is there a baseball in this image?


Now complete the following: 
[Text]:
{sent}
[Entity]:
{entity}
[Response]:
"""

PROMPT_TEMPLATE_REFINE_ENTITY = """
You are given a query, a passage and supplementary information.
[Task]
You are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. Correct the sentences in the passage if they are inconsistent with the supplementary information. Remove the objects that are confirmed to not exist in the supplementary information.
2. Do not modify correct sentences and introduce additional information.
3. If the sentence contains non-existent entities, delete the entire sentence.
4. The refined passage should be a reasonable answer to the query.
5. Note the dining table is equivalent to the table.
Output only the corrected passage, without introducing extra contents.

Here are examples:
[Query]:
Please describe this image in detail.
[Passage]:
There is a snowboard in the image. The image shows a person skiing down a snow-covered slope.
[Supplementary Information]:
right: There is a snowboard. 
right: There is a person.
wrong: There is a person skiing down a snow-covered slope.
[Response]: 
There is a snowboard in the image. The image shows a person.
[Query]:
Please describe this image in detail.
[Passage]:
The image features a brown dog, leaping into the air to catch a frisbee. There are several other people in the scene, with some standing closer to the dog and others further away.
[Supplementary Information]:
right: There is no people.
right: There is a dog.
right: There is a frisbee.
[Response]:
The image captures a brown dog leaping into the air to catch a frisbee.

Now complete the following:
[Query]:
Please describe this image in detail.
[Passage]:
{passage}
[Supplementary Information]:
{sup_info}
[Response]:
"""


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True




def get_gpt4v_answer(prompt, image_path):
    client = OpenAI(
      api_key="sk-xxxxxxxxxxxxxxxx"
    )
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    while True:
        # print("Cnt: ", cnt)
        try:
            response =  client.chat.completions.create(
            
                model='gpt-4-vision-preview',
                messages= [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ],
                max_tokens=300,
            )
            break
        except Exception as e:
            print(e)

        cnt += 1
        if cnt > 3:
            print("Network Error!")

    return response.choices[0].message.content


def get_lvlm_output(prompts, image, max_new_tokens=32):
    # print(prompts)
    with torch.inference_mode():
        with torch.no_grad():
            res = model.generate(
                {"image": norm(image), "prompt":prompts}, 
                use_nucleus_sampling=False, 
                num_beams=5,
                max_new_tokens=32,  #llava: 64
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
    sentence = res[0]
    return sentence

def get_mllm_answers(sample,image,question_key="entity_questions", answer_key="answers", additional_sample_counts=0, history=None):
    """
        Generate mllm answers to the questions.
        question_key = ["entity_questions", "generated_torture_questions"]
    """
    for dialogue in sample['dialogues']:
        # print("dialogue:",dialogue)
        gen_qs = dialogue[question_key]
        cur_answers = []
        if question_key=="entity_questions":
            qs=gen_qs
            if sample['has_yes']==True:
                answer='yes'
            else:
                answer='no'
            cur_answers.append(answer.strip())

        else:
            for cur_qs in gen_qs:
                qs, entity = cur_qs                   
                if qs==sample['query']:
                    if sample['has_yes']==True:
                        answer='yes'
                    else:
                        answer='no'
                else:    
                    print("qs:",qs)
                    prompts = [PROMPT_HISTORY_TEMPLATE_MINIGPT4.format(question=qs)]
                    answer = get_lvlm_output(prompts, image)
                cur_answers.append(answer.strip())
        dialogue[answer_key] = cur_answers
    return sample

def get_split_sents(doc):
        split_sents = list(doc.sents)
        split_sents = [sent.text.strip() for sent in split_sents]
        return split_sents

def remove_duplicates(res):
    qs_set = set()
    output = []
    for s in res:
        qs, ent = s
        if qs in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(qs)
    return output

def remove_duplicates_answer(res):
    qs_set = set()
    output = []
    for s in res:
        if s in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(s)
    return output

def remove_duplicates_wrong(res):
    se_set = set()
    output = []
    for se in res:
        if "wrong: There is a " in se:
            se=se.replace("wrong: There is a ","right: There is no ")
        elif "wrong: There is an " in se:
            se=se.replace("wrong: There is an ","right: There is no ")
        if se in se_set:
            continue
        else:
            output.append(se)
            se_set.add(se)
    return output


def get_response(prompt, sys_prompt, temperature=0.2, max_tokens=1024, ):
    content = prompt
    cnt = 1
    while True:
        # print("Cnt: ", cnt)
        try:
            response =  client.chat.completions.create(
            
                model='gpt-3.5-turbo-0125',
                messages=[{
                    'role': 'system',
                    'content': sys_prompt,
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

        cnt += 1
        if cnt > 3:
            print("Network Error!")

    res = response.choices[0].message.content
    return res 

def get_target(sample):
    """
    extract target entities in each sentence
    """
    extracted_entities = []
    set_entities = set()                                                                    # merge the same in differen sents
    entity_answer=""
    for a in sample["input_desc"]:
        entity_answer=entity_answer+a
    prompt = PROMPT_TEMPLATE_TARGET.format(sent=entity_answer)
    entity_str = get_response(prompt, sys_prompt=SYS_PROMPT_EXTRACTOR)
    entity_str = entity_str.strip().split('.')                                          
    entity_str = [item for item in entity_str if item != ""]                            
    extracted_entities.append(entity_str)                                              
    [set_entities.add(item) for item in entity_str]
    
    sample["split_sent_entity"] = extracted_entities
    sample["dialogues"] = [{"entity": entity} for entity in set_entities]

    
    return sample

def get_question_torture(sample):
    """
        Generate torture questions about the entities in the image
    """
    for dialogue in sample['dialogues']:
        # print('-'*50)
        entity = dialogue["entity"]

        cur_torture = []
        
        prompt_torture = PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=sample['input_desc'], entity=entity)
        torture_qs = get_response(prompt_torture, sys_prompt=SYS_PROMPT_EXTRACTOR)
        torture_qs = torture_qs.splitlines()
        torture_qs = [item.split('&') for item in torture_qs if item.lower() != 'none']                               
        torture_qs = [[item[1], item[0]] for item in torture_qs if len(item)==2]      # reverse, (question, answer)

        cur_torture.extend(torture_qs)
        cur_torture = remove_duplicates(cur_torture)    
        if len(cur_torture) > 5:                        # at least 5 questions
            cur_torture = cur_torture[:5]

        
        dialogue["entity_questions"]=sample["query"]
        dialogue["generated_torture_questions"] = cur_torture
        # print(dialogue)
        
    return sample

def judge(sample,clip_scorer,image):
    clip_threshold = 0.075
    yes_threshold = 0.5
    clip_scores=[]
    support_information=[]
    for dialogue in sample['dialogues']:
        id=0
        num_yes=0
        sup_info=[]
        for torture in dialogue["answers_torture"]:            
            t=torture.lower()
            t=t.replace('.', '')
            t=t.replace(',', '')
            t=t.split(' ')
            if 'no' in t:
                tor_judge='no'
            elif 'yes' in t:
                tor_judge='yes'
            elif dialogue['entity'] in torture and len(dialogue['entity'].split())==2:
                tor_judge='yes'
            elif dialogue['entity'] in t and len(dialogue['entity'].split())==1:
                tor_judge='yes'
            else:
                tor_judge='no'
            qs,se=dialogue["generated_torture_questions"][id]
            if tor_judge=='yes':
                sup_info.append("right: "+se)
                num_yes+=1
            else:
                s=se.lower()
                s=s.replace('.', '')
                s=s.replace(',', '')
                s=s.split(' ')
                if 'no' in s:
                    sup_info.append("right: "+se)
                    num_yes+=1
                else:
                    score=clip_scorer.get_clip_score(se, image)
                    clip_scores.append(dialogue["entity"]+str(score))
                    if score>=clip_threshold:
                        sup_info.append("right: "+se)
                        num_yes+=1
                    else:
                        sup_info.append("wrong: "+se)
            id=id+1
        r = num_yes/len(dialogue["answers_torture"])
        if r<yes_threshold: 
            sup_info=["right: There is no "+dialogue["entity"]+" . r="+str(r)]
            continue
        if r==1: 
            sup_info.append("right: There is a "+dialogue["entity"])
        support_information.extend(sup_info)

    support_information = remove_duplicates_wrong(support_information)    
    sample['clip_scores']=clip_scores
    sample['sup_info']=support_information
    return sample
                    


def get_refinement(sample):
    passage = sample['input_desc'][0]
    question=sample['query']
    if sample['sup_info']==[]:
        sample["output"] = passage
    else:
        support_information="\n".join(sample['sup_info'])+ "\n"
        prompt = PROMPT_TEMPLATE_REFINE_ENTITY.format(query=question, passage=passage, sup_info=support_information)
        refiend_passage = get_response(prompt, sys_prompt=SYS_PROMPT_REFINER)
        sample["output"] = refiend_passage.strip()
    return sample


parser = argparse.ArgumentParser(description="GPT-4v evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="/your/path/to/coco/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
args = parser.parse_known_args()[0]



os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)




img_files = os.listdir(args.data_path)
random.shuffle(img_files)

base_path = "log/gpt4v-eval"
if not os.path.exists(base_path):
    os.mkdir(base_path)
if not os.path.exists(base_path + f"/{args.model}"):
    os.mkdir(base_path + f"/{args.model}")

gpt_answer_records = {}
assistant_answer_records = {}
avg_hal_score_1 = 0
avg_hal_score_2 = 0
avg_det_score_1 = 0
avg_det_score_2 = 0
num_count = 0

clip_scorer = CLIPModel(device=device)

for idx in range(50):
    img = img_files[idx]
    image_path = args.data_path + img
    raw_image = Image.open(image_path)
    raw_image = raw_image.convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    qu = "Please describe this image in detail."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)
    assistant_answer_records[str(img)] = {}

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt":qu}, 
                use_nucleus_sampling=False, 
                num_beams=5,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
    model_response_1 = out[0]
    assistant_answer_records[str(img)]["assistant_1"] = model_response_1
    print("OPERA output:") 
    print(model_response_1)


    claim=out
    sample = {  
        'query': qu,
        'img_path': image_path,
        'input_desc': claim,
    }                
    sample['has_yes']=True 
    sample = get_target(sample)                   
    sample = get_question_torture(sample)
    sample = get_mllm_answers(sample, image, question_key="entity_questions", answer_key="answers", additional_sample_counts=2,history=None)
    sample = get_mllm_answers(sample, image, question_key="generated_torture_questions", answer_key="answers_torture", additional_sample_counts=0,history=1)
    print("get mllm answers finish!")
    sample = judge(sample,clip_scorer,raw_image)
    print("sup_info:",sample["sup_info"])
    sample = get_refinement(sample) 


    model_response_2 = sample["output"]
    assistant_answer_records[str(img)]["assistant_2"] = model_response_2
    print("Logic check output:")
    print(model_response_2)

    # gpt-4v eval
    prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)

    gpt_answer = get_gpt4v_answer(prompt, image_path)
    print(gpt_answer)
    gpt_answer_records[str(img)] = gpt_answer
    print(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" "))
    print(len(gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")))
    try:
        hal_score_1, hal_score_2 = gpt_answer.split("Accuracy: ")[-1].split("\n")[0].split(" ")
        det_score_1, det_score_2 = gpt_answer.split("Detailedness: ")[-1].split("\n")[0].split(" ")
    except:
        continue
    avg_hal_score_1 += int(hal_score_1)
    avg_hal_score_2 += int(hal_score_2)
    avg_det_score_1 += int(det_score_1)
    avg_det_score_2 += int(det_score_2)
    num_count += 1
    print("=========================================")

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'answers.json'), "w") as f:
        json.dump(assistant_answer_records, f)

    # dump metric file
    with open(os.path.join(base_path + f"/{args.model}", 'records.json'), "w") as f:
        json.dump(gpt_answer_records, f)

avg_score = float(avg_hal_score_1) / num_count
avg_score = float(avg_hal_score_2) / num_count
avg_score = float(avg_det_score_1) / num_count
avg_score = float(avg_det_score_2) / num_count
print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_hal_score_1}; {avg_hal_score_2}")
print(f"The avg det score for Assistant 1 and Assistent 2: {avg_det_score_1}; {avg_det_score_2}")