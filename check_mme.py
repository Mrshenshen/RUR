import argparse
import os
import random
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

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

from PIL import Image
from torchvision.utils import save_image
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json
import time
from lib.clip_utils import CLIPModel


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
print("Load LLM Done.")

parser = argparse.ArgumentParser(description="MME evaluation on LVLMs.")
parser.add_argument("--model", type=str, default='llava-1.5', help="model")
parser.add_argument("--gpu-id", type=str, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--type", type=str, default="position", help="mme type, like color, count, existence, position")
parser.add_argument("--data_path", type=str, default="/your/path/to/MME_Benchmark_release_version/", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--beam", type=int,default=5)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
args = parser.parse_known_args()[0]


# --------------------- prompt design ---------------------
SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."

PROMPT_TEMPLATE_LLAVA_EXISTENCE = '''
USER:
<ImageHere>
Just base on the information in the image. Find evidence and answer: {question} 
Provide as much detailed information as possible, including its attributes, location, or state.
ASSISTANT:
'''
PROMPT_TEMPLATE_LLAVA_POSITION = '''
USER:
<ImageHere> 
{question1}
ASSISTANT:
'''
PROMPT_TEMPLATE_LLAVA_COUNT = '''
USER:
<ImageHere>
How many {entity} are there in the picture? {question}
Provide as much detailed information as possible, including its quantity, attributes, or state.
ASSISTANT:
'''

PROMPT_TEMPLATE_INSTRUCTBLIP_EXISTENCE = '''
<ImageHere>
Just base on the information in the image. Find evidence and answer: {question} 
Provide as much detailed information as possible, including its attributes, location, or state.
'''
PROMPT_TEMPLATE_INSTRUCTBLIP_POSITION = '''
<ImageHere>
{question1} {question2}
'''
PROMPT_TEMPLATE_INSTRUCTBLIP_COUNT = '''
<ImageHere>
How many {entity} are there in the picture? {question}
Provide as much detailed information as possible, including its quantity, attributes, or state.
'''
PROMPT_TEMPLATE_INSTRUCTBLIP_COLOR = '''
<ImageHere>
What color is the {entity} in the picture?
Provide as much detailed information as possible, including its quantity, attributes, or state.
'''
PROMPT_TEMPLATE_MINIGPT4_EXISTENCE = '''
###Human: 
<Img><ImageHere></Img>
Just base on the information in the image. Find evidence and answer: {question} 
Provide as much detailed information as possible, including its attributes, location, or state.
###Assistant:
'''
PROMPT_TEMPLATE_MINIGPT4_POSITION = '''
###Human: 
<Img><ImageHere></Img>
{question1} {question2}
###Assistant:
'''
PROMPT_TEMPLATE_MINIGPT4_COUNT = '''
###Human: 
<Img><ImageHere></Img>
How many {entity} are there in the picture? {question}
Provide as much detailed information as possible, including its quantity, attributes, or state.
###Assistant:
'''
PROMPT_TEMPLATE_MINIGPT4_COLOR = '''
###Human: 
<Img><ImageHere></Img>
What color is the {entity} in the picture?
Provide as much detailed information as possible, including its quantity, attributes, or state.
###Assistant:
'''

PROMPT_TEMPLATE_TARGET = """"
You are given a sentence, extract the entities within the sentence for me. 
[Task]
Your task is to extract the common objects and summarize them as general categories without repetition, merging essentially similar objects.
Avoid extracting abstract or non-specific entities. 
Extract entity in the singular form. 
Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None".
DO NOT RESPOND WITH ANYTHING ELSE.


Here are examples:
[Sentence]:
Are there a total of two trains in the picture?
[Response]:
train
[Sentence]:
Is there a man wearing a white shirt in the image?
[Response]:
man.shirt
[Sentence]:
Is the blue umbrella under the black umbrella?
[Response]:
umbrella
[Sentence]:
Is the car on the left side of the fire hydrant in the picture?
[Response]:
car.fire hydrant


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
3. The refined passage should be a reasonable answer to the query. The refined passage should have 'Yes' or 'No' at the beginning.
4. If supplementary information and passage don't mention the entity in the query, you should answer no to the query.
5. Note the dining table is equivalent to the table.
6. Note the mannequins is not people.
Output only the corrected passage, without introducing extra contents.

Here are examples:
[Query]:
Is there a red boat in the image?
[Passage]:
Yes, there is a red boat in the image. It is a small red boat, possibly a fishing boat, and it is docked at a pier.
[Supplementary Information]:
right: The boat is red.
right: The boat is small.
wrong: The boat is docked at a pier.
right: There is a red boat in the image.
right: The red boat is small.
[Response]: 
Yes, there is a red boat in the image. It is a small red boat, possibly a fishing boat.
[Query]:
Is there a chair in the image?
[Passage]:
No, there is no chair in the image. There is a sports ball in the image, and it appears to be a soccer ball.
[Supplementary Information]:
right: There is a sports ball in the image.
right: There is no chair.
[Response]: 
No, there is no chair in the image. And there is a sports ball in the image.
[Query]:
Is there a ball in the image?
[Passage]:
There is a person in this image.
[Supplementary Information]:
right: There is a person in this image.
[Response]: 
No, there is no ball in the image, and there is a person in this image.

Now complete the following:
[Query]:
{query}
[Passage]:
{passage}
[Supplementary Information]:
{sup_info}
[Response]:
"""

PROMPT_TEMPLATE_REFINE_ANSWER = """
You are given a query and a passage.
[Task]
You need to read the passage and answer the query based on the passage, following these rules:
1. Answer query using only 'Yes' or 'No' based on the content of the passage.
2. Read the passage and understand it's information. The passage contains a description of an image, and its content serves as the basis for answering query.
3. Pay attention to whether the attributes of the entity are correct, including color, quantity, position, etc.
Only output yes or no, do not output any other content.

Here are examples:
[Query]:
Is there a red boat in the image?
[Passage]:
There is a red boat in the image. It is a small red boat, possibly a fishing boat, and it is docked at a pier.
[Response]: 
Yes
[Query]:
Is the dog above the pool in the image?
[Passage]:
In the image, a black and white dog is jumping above a swimming pool. The dog is in mid-air, showcasing its agility and athleticism.
[Response]: 
Yes
[Query]:
Is there a brown cat in the image?
[Passage]:
In the image, there is a black cat standing next to a white sink.
[Response]: 
No
[Query]:
Are there two horses in this image?
[Passage]:
In the image, there is a man standing next to a horse that is pulling a plow. There is another horse in the background.
[Response]: 
Yes

Now complete the following:
[Query]:
{query}
[Passage]:
{passage}
[Response]:
"""

def get_lvlm_output(image,prompts):
    with torch.no_grad():
        res = model.generate(
            {"image": norm(image), "prompt":prompts}, 
            use_nucleus_sampling=args.sample, 
            num_beams=args.beam,
            max_new_tokens=64,
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
                    template = INSTRUCTION_TEMPLATE[args.model]
                    prompts = [template.replace("<question>", qs)]
                    answer = get_lvlm_output(image, prompts)
                cur_answers.append(answer.strip())

        dialogue[answer_key] = cur_answers
    return sample

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


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
        elif "wrong: There is " in se:
            se=se.replace("wrong: There is ","right: There is no ")
        elif "wrong: There are " in se:
            se=se.replace("wrong: There are ","right: There are no ")
        if se in se_set:
            continue
        else:
            output.append(se)
            se_set.add(se)
    return output

def remove_wrong(res):
    se_set = set()
    output = []
    for se in res:
        if "wrong:" in se:
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
            
                model='gpt-3.5-turbo-0125',#'gpt-3.5-turbo-0125'
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

def get_response_gpt4(prompt, sys_prompt, temperature=0.2, max_tokens=1024, ):
    content = prompt
    cnt = 1
    while True:
        # print("Cnt: ", cnt)
        try:
            response =  client.chat.completions.create(
            
                model='gpt-4-turbo-preview',
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
    set_entities = set()                                                                  
    prompt = PROMPT_TEMPLATE_TARGET.format(sent=sample['query'])
    entity_str = get_response(prompt, sys_prompt=SYS_PROMPT_EXTRACTOR)
    entity_str = entity_str.strip().split('.')                                          
    entity_str = [item for item in entity_str if item != "" and item != 'None']         # delete ""
    extracted_entities.append(entity_str)                                               
    [set_entities.add(item) for item in entity_str]
    
    # print("Entity: ", set_entities)
    sample["split_sent_entity"] = extracted_entities
    sample["dialogues"] = [{"entity": entity} for entity in set_entities]
    # print("sample after get target:",sample)
    
    return sample

def get_question_torture(sample):
    """
        Generate torture questions about the entities in the image
    """
    for dialogue in sample['dialogues']:
        # print('-'*50)
        entity = dialogue["entity"]
        cur_torture = []
        ans=""
        if sample['has_yes']==True:
            for y in sample['yes_answers']:
                ans=ans+y
        else:
            for y in sample['input_desc']:
                ans=ans+y
        prompt_torture = PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=ans, entity=entity)
        torture_qs = get_response(prompt_torture, sys_prompt=SYS_PROMPT_EXTRACTOR)
        torture_qs = torture_qs.splitlines()
        torture_qs = [item.split('&') for item in torture_qs if item.lower() != 'none']                               
        torture_qs = [[item[1], item[0]] for item in torture_qs if len(item)==2]      # reverse, (question, answer)
        cur_torture.extend(torture_qs)
        cur_torture = remove_duplicates(cur_torture)    
        if len(cur_torture) > 5:                        # at least 5 questions
            # torture_qs = torture_qs[:10]
            dialogue["entity_questions"]=sample["query"]
            dialogue["generated_torture_questions"] = cur_torture
            continue

        
        dialogue["entity_questions"]=sample["query"]
        dialogue["generated_torture_questions"] = cur_torture
        # print(dialogue)
        
    return sample

def judge(sample,clip_scorer,image):
    sup_info=[]
    if args.type == 'existence':
        threshold = 0.12   
    elif args.type == 'color':
        threshold = 0.12
    elif args.type == 'position':
        threshold = 0.09
    else:
        threshold = 0.075
    clip_scores=[]
    for dialogue in sample['dialogues']:
        id=0
        for torture in dialogue["answers_torture"]:
            t=torture.lower()
            t=t.replace('.', '')
            t=t.replace(',', '')
            t=t.split(' ')
            if 'no' in t:
                tor_judge='no'
            elif 'yes' in t:
                tor_judge='yes'
            elif 'not' in t:
                tor_judge='no'
            elif args.type=="existence":            
                if dialogue['entity'] in torture and len(dialogue['entity'].split())==2:
                    tor_judge='yes'
                elif dialogue['entity'] in t and len(dialogue['entity'].split())==1:
                    tor_judge='yes'
                else:
                    tor_judge='no'
            else:
                tor_judge='no'
            qs,se=dialogue["generated_torture_questions"][id]
            if tor_judge=='yes':
                sup_info.append("right: "+se)
            elif qs==sample['query']:            
                t = sample['query'].strip().split()
                if len(t)==7:
                    entity=t[3]
                elif len(t)==8:
                    entity=t[3]+" "+t[4]
                if dialogue['answers']=='no':        
                    sup_info.append("right: There is no "+entity+" .")
                elif dialogue['answers']=='yes':
                    sup_info.append("right: There is a "+entity+" .")
            else:
                s=se.lower()
                s=s.replace('.', '')
                s=s.replace(',', '')
                s=s.split(' ')
                if 'no' in s:
                    sup_info.append("right: "+se)
                else:
                    score=clip_scorer.get_clip_score(se, image)
                    # print("score:",score)
                    clip_scores.append(dialogue["entity"]+str(score))
                    if score>=threshold:
                        sup_info.append("right: "+se)
                    else:
                        sup_info.append("wrong: "+se)
            id=id+1
    sup_info = remove_duplicates_wrong(sup_info)    # remove duplicates and turn wrong：There is a/an into right:There is no
    if args.type=="existence":
        sup_info = remove_wrong(sup_info)    # remove wrong

    sample['clip_scores']=clip_scores
    sample['sup_info']=sup_info
    return sample
                    


def get_refinement(sample):
    threshold=0.5
    if args.type=="existence":
        threshold=0.49
    if sample['has_yes']==True:
        passage = sample['yes_answers'][0]
    else:
        passage = sample['input_desc'][0]
    question=sample['query']

    if sample['sup_info']==[]:
        sample["output"] = sample["input_desc"][0]
    else:
        nli_value = 0 
        predict_labels=sample['sup_info']
        for label in predict_labels:
            if "right" in label:
                nli_value += 1
        print("nli_value:",nli_value)
        r = nli_value/len(predict_labels) if len(predict_labels) > 0 else 0
        print("r:",r)
        if sample['has_yes']==True:
            entity_answer='yes'
        else:
            entity_answer='no'
        if r <= threshold and entity_answer=='yes' and len(predict_labels)>2:
            sample["output"] = 'No, for wrong/len > threshold.'
        elif r==1 and entity_answer=='yes':
            sample['output'] = sample['input_desc'][0]
        else:
            sup_info="\n".join(sample['sup_info'])+ "\n"
            prompt = PROMPT_TEMPLATE_REFINE_ENTITY.format(query=question, passage=passage, sup_info=sup_info)
            refiend_passage = get_response(prompt, sys_prompt=SYS_PROMPT_REFINER)
            sample["output"] = refiend_passage.strip()
        if 'no' in sample["query"]:
            if 'No' in sample["output"]:
                sample["output"]=sample["output"].replace("No","Yes")    
            elif 'Yes' in sample["output"]:
                sample["output"]=sample["output"].replace("Yes","No")                    
    return sample

def get_yes_or_no(sample):
    prompt = PROMPT_TEMPLATE_REFINE_ANSWER.format(query=sample['query'],passage=sample['output'])
    final_answer = get_response_gpt4(prompt, sys_prompt=SYS_PROMPT_REFINER)
    sample["final"] = final_answer.strip()
    return sample



def keep_before_last_period(sentence):  
    last_period_index = sentence.rfind('.')  
    if last_period_index != -1:  
        return sentence[:last_period_index+1]  
    else:  
        return sentence  

def only_ask(sentence):    #Keep the question part in the sentence of mme and delete "Please answer yes or no."
    sentence = sentence.rstrip()  
    if sentence.endswith("Please answer yes or no."):  
        sentence = sentence[:-len("Please answer yes or no.")]  
    return sentence





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

img_files = []

data_path=args.data_path+args.type+"/"
# read in all the images in a folder
for file in os.listdir(data_path):
    if file.endswith(".jpg"):
        img_files.append(file)

print("img_files", len(img_files))

base_dir  = "./log/check_mme/"+args.model   # save answer
if not os.path.exists("./log/check_mme"):
    os.mkdir("./log/check_mme")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

torture_log = "./mme_samples/mme_"+args.model+"/"+args.type+"/%d.json"   #save samples for every question
if not os.path.exists("./mme_samples/mme_"+args.model):
    os.mkdir("./mme_samples/mme_"+args.model)
if not os.path.exists("./mme_samples/mme_"+args.model+"/"+args.type):
    os.mkdir("./mme_samples/mme_"+args.model+"/"+args.type)

clip_scorer = CLIPModel(device=device)


iterations = 2*len(img_files)

result_txt = []

prompts = []

if args.model=="llava-1.5":
    prompts=[PROMPT_TEMPLATE_LLAVA_EXISTENCE,PROMPT_TEMPLATE_LLAVA_POSITION,PROMPT_TEMPLATE_LLAVA_COUNT,PROMPT_TEMPLATE_LLAVA_EXISTENCE]
elif args.model=="instructblip":
    prompts=[PROMPT_TEMPLATE_INSTRUCTBLIP_EXISTENCE,PROMPT_TEMPLATE_INSTRUCTBLIP_POSITION,PROMPT_TEMPLATE_INSTRUCTBLIP_COUNT,PROMPT_TEMPLATE_INSTRUCTBLIP_COLOR]
elif args.model=="minigpt4":
    prompts=[PROMPT_TEMPLATE_MINIGPT4_EXISTENCE,PROMPT_TEMPLATE_MINIGPT4_POSITION,PROMPT_TEMPLATE_MINIGPT4_COUNT,PROMPT_TEMPLATE_MINIGPT4_COLOR]

id=1
for idx in tqdm(range(iterations)):
    new_line = ""
    img_file = img_files[int(idx/2)]
    # if id <10:
    #     id+=1
    #     continue
    print("id:",id)
    id+=1
    new_line += img_file + "\t"
    print("img_file", img_file)
    txt_file = img_file.replace(".jpg", ".txt")
    # get the first line of the txt file
    if idx % 2 == 0:
        with open(data_path + txt_file, "r") as f:
            qu = f.readlines()[0]
            if "Yes" in qu:
                gt = "Yes"
            else:
                gt = "No"
            qu = qu.replace("Yes", "")
            qu = qu.replace("No", "")

        print("idx % 2 == 0", qu)
    else:
    # get the second line of the txt file
        with open(data_path + txt_file, "r") as f:
            qu = f.readlines()[1]
            if "Yes" in qu:
                gt = "Yes"
            else:
                gt = "No"
            qu = qu.replace("Yes", "")
            qu = qu.replace("No", "")
        print("idx % 2 == 1", qu)

    new_line += qu + "\t" + gt + "\t"
    print("label:",gt)
    img_id = int(img_file.split(".jpg")[0][-6:])

    img_save = {}
    img_save["id"] = id
    img_save["image_id"] = img_id

    image_path = data_path + img_file
    raw_image = Image.open(image_path).convert('RGB')

    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    qu = only_ask(qu)
    sample = {  
        'query': qu,
        'img_path': image_path,
    }                
    sample = get_target(sample)
    entity = sample["split_sent_entity"][0]
    print("entity:",entity)
    if args.type=='position':
        if len(entity)==2:
            question="Which position is the "+entity[0]+" in relation to the "+entity[1]+" in the image? Pay attention to the positional relationship of them in the image. "+qu
        elif len(entity)==1:
            question="What is the positional relationship between the two "+entity[0]+" in the image? Use colors to distinguish them, pay attention to the positional relationship of them in the image. "
        question = prompts[1].format(question1=question)
    elif args.type=='count':
        question = prompts[2].format(entity=entity[0],question=qu)
    elif args.type=='color':
        question = prompts[3].format(entity=entity[0])
    elif args.type=='existence':
        question = prompts[0].format(question=qu)
    else:
        print("Wrong type of MME!!!")
        break
    print("question",question)
    claim=[]
    out = get_lvlm_output(image,question)
    claim.append(out.strip())
    print("opera output:") 
    for c in claim:
        print(c)
    sample['input_desc']=claim
    yes_answer=[]
    for answer in claim:
        entity_answer=answer.lower()
        entity_answer=entity_answer.replace('.', '')
        entity_answer=entity_answer.replace(',', '')
        entity_answer=entity_answer.split(' ')    
        if 'yes' in entity_answer:
            yes_answer.append(answer)
        elif 'no' in entity_answer:
            continue
        else:
            yes_entity=0
            for e in entity:#If there is entity in a sentence which doesn't contant yes or no, we believe it is yes.
                if e in answer and len(e.split())>1:
                    yes_entity+=1
                elif e in entity_answer and len(e.split())==1:
                    yes_entity+=1
            if yes_entity==len(entity):
                yes_answer.append(answer)
                

    sample['yes_answers']=yes_answer
    if yes_answer==[]:
        sample['has_yes']=False
    else:
        sample['has_yes']=True
    sample["output"]=claim[0]
    print("output:",sample["output"])
    img_save["caption"] = sample["output"]
    s=sample["output"].lower()
    s=s.replace('.', '')
    s=s.replace(',', '')
    s=s.split(' ')

    if 'yes' in s:
        answer='Yes'
    elif 'no' in s:
        answer='No'
    else:       
        sample=get_yes_or_no(sample)
        answer=sample["final"]
    print("final_answer:",answer)

   

    with open(torture_log % (id-1), 'w') as f:
        json_sample = json.dumps(sample, indent=4)
        f.write(json_sample+"\n")

    generated_captions_path = os.path.join(
        base_dir,
        f"{args.type}.json",
    )
    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")


    new_line += answer
    new_line = new_line.replace("\n", "")
    new_line = new_line.replace("\t\t", "\t")
    new_line += "\n"
    print({"new line":new_line})
    result_txt.append(new_line)
    with open(generated_captions_path.replace(".json", ".txt"), "w") as f:
        f.writelines(result_txt)
    


