import argparse
import os
import random

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


# --------------------- prompt design ---------------------

PROMPT_HISTORY_TEMPLATE_LLAVA = """
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, 
and polite answers to the human's questions and briefly describes the characteristics of the entity.
USER:
<ImageHere>
{question}
Provide as much detailed information as possible.
ASSISTANT:
"""

PROMPT_HISTORY_TEMPLATE_INSTRUCTBLIP= """
You are an image analyst. You need to observe the image carefully and answer question.
<ImageHere>{question} 
"""

PROMPT_HISTORY_TEMPLATE_MINIGPT4= """
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, 
and polite answers to the human's questions and briefly describes the characteristics of the entity.
###Human: 
<Img><ImageHere></Img> {question} Provide as much detailed information as possible.
###Assistant:
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


def get_lvlm_output(image, prompts,max_new_tokens=64):
    
    with torch.no_grad():
        res = model.generate(
            {"image": norm(image), "prompt":prompts}, 
            use_nucleus_sampling=args.sample, 
            num_beams=args.beam,
            max_new_tokens=max_new_tokens,  
            output_attentions=True,
            opera_decoding=True,
            scale_factor=args.scale_factor,
            threshold=args.threshold,
            num_attn_candidates=args.num_attn_candidates,
            penalty_weights=args.penalty_weights,
        )
    sentence = res[0]
    return sentence

def get_mllm_answers(sample,image,question_key="entity_questions", answer_key="answers", history_template=None):
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
                if qs==sample['query']: # When attribute question=entity question, directly use the answer of the entity question
                    if sample['has_yes']==True:
                        answer='yes'
                    else:
                        answer='no'
                else:    
                    prompts = [history_template.format(question=qs)]
                    answer = get_lvlm_output(image, prompts, 32)
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





parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, default='minigpt4', help="model")
parser.add_argument("--gpu-id", type=str, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="/your/path/to/coco/val2014/", help="coco data path")
parser.add_argument("--num_img", type=str, default=500, help="Number of images")

parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--beam", type=int,default=5)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)

args = parser.parse_known_args()[0]

SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."

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
    set_entities = set()                                                                   
    
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
        

        for ans in sample['input_desc']:
            prompt_torture = PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=ans, entity=entity)
            torture_qs = get_response(prompt_torture, sys_prompt=SYS_PROMPT_EXTRACTOR)
            torture_qs = torture_qs.splitlines()
            torture_qs = [item.split('&') for item in torture_qs if item.lower() != 'none']                               
            torture_qs = [[item[1], item[0]] for item in torture_qs if len(item)==2]      # reverse, (question, answer)

            cur_torture.extend(torture_qs)
            cur_torture = remove_duplicates(cur_torture)    # Remove duplicate sentences
            if len(cur_torture) > 5:                        # at least 5 questions
                cur_torture=cur_torture[:5]
                break

        
        dialogue["entity_questions"]=sample["query"]
        dialogue["generated_torture_questions"] = cur_torture
        # print(dialogue)
        
    return sample

def judge(sample,clip_scorer,image):
    clip_threshold = 0.055   
    yes_threshold = 0.4 
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
                    # print("clip score:",score)
                    clip_scores.append(dialogue["entity"]+str(score))
                    if score>=clip_threshold:
                        sup_info.append("right: "+se)
                        num_yes+=1
                    else:
                        sup_info.append("wrong: "+se)
            id=id+1
        r = num_yes/len(dialogue["answers_torture"])
        if r<yes_threshold: # If r<yes_threshold, it is considered that the entity does not exist.
            sup_info=["right: There is no "+dialogue["entity"]+" . r="+str(r)]
            continue
        if r==1: # If r=1, we believe that the entity must exist
            sup_info.append("right: There is a "+dialogue["entity"])
        support_information.extend(sup_info)

    support_information = remove_duplicates_wrong(support_information)    # remove duplicates and turn wrong：There is a/an into right:There is no
    sample['clip_scores']=clip_scores
    sample['sup_info']=support_information
    return sample
                    


def get_refinement(sample):
    passage = sample['input_desc'][0]
    question = sample['query']
    if sample['sup_info']==[]:
        sample["output"] = passage
    else:
        support_information="\n".join(sample['sup_info'])+ "\n"
        prompt = PROMPT_TEMPLATE_REFINE_ENTITY.format(query=question, passage=passage, sup_info=support_information)
        refiend_passage = get_response(prompt, sys_prompt=SYS_PROMPT_REFINER)
        sample["output"] = refiend_passage.strip()
    return sample

def keep_before_last_period(sentence):  
    last_period_index = sentence.rfind('.')  
    if last_period_index != -1:  
        return sentence[:last_period_index+1]  
    else:  
        return sentence  

def sort_and_load_images(folder_path, max_images=500):  
    images = []  
    for filename in os.listdir(folder_path):   
        file_path = os.path.join(folder_path, filename)  
        images.append(file_path)  
  
    images.sort()  
    selected_images = images[:min(max_images, len(images))]   
    return selected_images  



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


img_path=args.data_path
num_img=args.num_img
img_path=sort_and_load_images(img_path,num_img)
progress_bar = tqdm(total=num_img, desc='Processing')
base_dir  = "./log/check_opera/" + args.model   # Store answers
if not os.path.exists("./log/check_opera"):
    os.mkdir("./log/check_opera")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
torture_log = "./chair_samples/chair_sample_"+args.model+"/%d.json" # Store samples for every image
if not os.path.exists("./chair_samples"):
    os.mkdir("./chair_samples")
if not os.path.exists("./chair_samples/chair_sample_"+args.model):
    os.mkdir("./chair_samples/chair_sample_"+args.model)
clip_scorer = CLIPModel(device=device)
times=1
if args.model=="llava-1.5":
    first_question="Please describe this image in detail. Provide as much detailed information as possible, including its attributes, location, or state. Base on the image to describe."
    history_template=PROMPT_HISTORY_TEMPLATE_LLAVA
elif args.model=="instructblip":
    first_question="Please describe this image in detail."
    history_template=PROMPT_HISTORY_TEMPLATE_INSTRUCTBLIP
elif args.model=="minigpt4":
    first_question="Please describe this image in detail. Provide as much detailed information as possible, including its attributes, location, or state. Base on the image to describe."
    history_template=PROMPT_HISTORY_TEMPLATE_MINIGPT4


for img_file in os.listdir(img_path):
    if times>num_img:
        break
        
    times+=1
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_save = {}
    img_save["image_id"] = img_id
    image_path = img_file
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", first_question)
    claim=[]
    out = get_lvlm_output(image,qu,64)
    out = keep_before_last_period(out)  # Remove incomplete sentences at the end
    claim.append(out.strip())
    print("opera output:") 
    for c in claim:
        print(c)
    sample = {  
        'query': first_question,
        'img_path': image_path,
        'input_desc': claim,
    }                
    sample['has_yes']=True  # By default, we believe all entities in CHAIR exist
    sample = get_target(sample)          # object extraction
    sample = get_question_torture(sample)# feature extraction & question
    sample = get_mllm_answers(sample, image, question_key="entity_questions", answer_key="answers", history_template=history_template)
    sample = get_mllm_answers(sample, image, question_key="generated_torture_questions", answer_key="answers_torture", history_template=history_template)
    print("get mllm answers finish!")
    sample = judge(sample,clip_scorer,raw_image)
    print("sup_info:",sample["sup_info"])
    sample = get_refinement(sample) 
    print("output:",sample["output"])
    img_save["answer"]=sample['output']

    
    with open(torture_log % (img_id), 'w') as f:
        json_sample = json.dumps(sample, indent=4)
        f.write(json_sample+"\n")
    with open(os.path.join(base_dir, 'chair_{}.jsonl'.format(args.model)), "a") as f:
        json.dump(img_save, f)
        f.write('\n')
    progress_bar.update()

progress_bar.close()