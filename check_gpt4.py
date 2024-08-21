import argparse
import os
from minigpt4.common.config import Config
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from openai import OpenAI

from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.models import load_preprocess
from PIL import Image
import tqdm
import time
from shr_eval.shr_utils import *
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


SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."


GPT_JUDGE_PROMPT = '''
Please help me judge if the comment of this image is hallucination or correct. 
I will give you a list of region description of a image. The format is [x1, y1, x2, y2]: region description, where [x1, y1, x2, y2] is the bounding box of the region. Highly overlapping bounding boxes may refer to the same object. This is the ground truth information of the image. Besides, I give you some factual information about the content of the image (which is 100% accurate). Your judgement should base on this information. However, this information only descibe the objects in the region of image, so it cannot descibe the subjective part of the image, e.g., atmosphere, style, emotion. In that case, you can return "Cannot judge".
Also, I will give you a list of comments of the image for you to judge if it is hallucination. Please give a judgement one by one along with the reason.

Your output should be:
Judgement:
1. hallucination or correct or cannot judge: <reason>
2. ...

Here are the region descriptions of the image:
{}

Factual Information:
{}

Here is the comment for you to judge (hallucination, correct, or cannot judge): 
{}
'''


client = OpenAI(
      api_key="sk-xxxxxxxxxxxxxxxx"
)
NUM_SECONDS_TO_SLEEP = 0.5

PROMPT_HISTORY_TEMPLATE_INSTRUCTBLIP= """
<ImageHere>{question} 
"""
PROMPT_HISTORY_TEMPLATE_MINIGPT4 = """
###Human: <Img><ImageHere></Img> {question} ###Assistant:
"""
PROMPT_HISTORY_TEMPLATE_LLAVA = """
###Human: <Img><ImageHere></Img> {question} ###Assistant:
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


def get_response_gpt4(prompt, model_name = "gpt-4o", temperature=0.7, max_tokens=800, ):
    content = prompt
    cnt = 1
    while True:
        # print("Cnt: ", cnt)
        try:
            response =  client.chat.completions.create(
                model=model_name,
                messages = [{"role": "user", "content": prompt}],
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
                top_p=1.0,
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


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_lvlm_output(image, prompts,max_new_tokens=12):
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
                    answer = get_lvlm_output(image,prompts)
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
    set_entities = set()                                                                    
    entity_answer=""
    for a in sample["input_desc"]:
        entity_answer=entity_answer+a
    prompt = PROMPT_TEMPLATE_TARGET.format(sent=entity_answer)
    entity_str = get_response(prompt, sys_prompt=SYS_PROMPT_EXTRACTOR)
    # print("target from gpt:",entity_str)
    entity_str = entity_str.strip().split('.')                                          
    entity_str = [item for item in entity_str if item != ""]                            
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

        prompt_torture = PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=sample['input_desc'], entity=entity)
        torture_qs = get_response(prompt_torture, sys_prompt=SYS_PROMPT_EXTRACTOR)
        torture_qs = torture_qs.splitlines()
        torture_qs = [item.split('&') for item in torture_qs if item.lower() != 'none']                               
        torture_qs = [[item[1], item[0]] for item in torture_qs if len(item)==2]      # reverse, (question, answer)

        cur_torture.extend(torture_qs)
        cur_torture = remove_duplicates(cur_torture)    
        if len(cur_torture) > 2:                        
            cur_torture = cur_torture[:2]

        
        dialogue["entity_questions"]=sample["query"]
        dialogue["generated_torture_questions"] = cur_torture
        # print(dialogue)
        
    return sample

def judge_sentence(sample,clip_scorer,image):
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
                    # print("score:",score)
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



def main(args):

    val_images = json.load(open(os.path.join(args.shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(args.vg_path, "image_data.json")))
    id2path = {
        _data["image_id"]:os.path.join(args.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]:_data for _data in vg_image_data}
    region = json.load(open(os.path.join(args.vg_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]:r for r in region}
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    
    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(args.shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(args.shr_path, "shr_factual_part2.jsonl")

    base_eval_path = "./log/gpt4/"+args.model
    if not os.path.exists("./log/gpt4"):
        os.mkdir("./log/gpt4")
    if not os.path.exists(base_eval_path):
        os.mkdir(base_eval_path)


    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    
    for _data in tqdm.tqdm(val_images):
        image_id = _data["image_id"]
        image_path = id2path[int(image_id)]
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)
        inp = "Describe this image in detail."
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", inp)
        outputs0=get_lvlm_output(image,qu,512)
        
        print("lvlm outputs:",outputs0)
        claim=outputs0
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
        sample = judge_sentence(sample,clip_scorer,raw_image)
        print("sup_info:",sample["sup_info"])
        if sample["sup_info"]==[]:
            sample["output"]=sample['input_desc'][0]
        else:
            sample = get_refinement(sample)
         
        print("logic outputs:",sample["output"])

        outputs = sample["output"]


        # get GPT judgement
        description = get_desc(id2img, id2reg, int(image_id))
        
        model_cap_sep, is_repeated = get_model_cap(outputs)
        print("model_cap_sep:",model_cap_sep)
        # calculate repetition
        gram1 = cal_repetition(outputs,1)
        gram2 = cal_repetition(outputs,2)
        gram3 = cal_repetition(outputs,3)
        gram4 = cal_repetition(outputs,4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4
            
        # skip gpt judgement 
        if args.no_gpt_judge:
            continue
            
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"
        # GPT judgement
        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
        if len(judge_prompt) > 15000:
            print(f"skip {image_id} for too long prompt!")
            continue
        
        
        for run in run_all:
            while True:
                # judge = get_gpt_response(prompt=judge_prompt)
                judge = get_response_gpt4(prompt=judge_prompt)
                print("judge:",judge)
                if "Judgement" not in judge:
                    print(f"No judgement found for {image_id}")
                    continue
                else:
                    break
            # post-process
            final_judge = post_process_no_revise(judge, outputs)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "model_response": outputs,
                "judgement": final_judge,
            }
        
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(_gram1/len(val_images), 3)}")
        print(f"gram-2 repetition: {round(_gram2/len(val_images), 3)}")
        print(f"gram-3 repetition: {round(_gram3/len(val_images), 3)}")
        print(f"gram-4 repetition: {round(_gram4/len(val_images), 3)}")
    else:
        localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
        if not os.path.exists(os.path.join(base_eval_path)):
            os.mkdir(os.path.join(base_eval_path))
        # dump config file
        eval_path = os.path.join(os.path.join(base_eval_path, localtime))
        os.mkdir(eval_path)
        # save metrics
        metrics = {}
        for run in run_all:
            metrics[run] = {}
            get_metric(judgement[run], metrics[run])
        # repetition
        metrics['gram-1-repetition'] = round(_gram1/len(val_images), 3)
        metrics['gram-2-repetition'] = round(_gram2/len(val_images), 3)
        metrics['gram-3-repetition'] = round(_gram3/len(val_images), 3)
        metrics['gram-4-repetition'] = round(_gram4/len(val_images), 3)
        # halucination ratio
        metrics["mean_hal_ratio"] = round(
            sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
        )
        metrics["model_base"] = args.model
        # dump judgement file
        with open(os.path.join(base_eval_path, localtime, 'judgement_check_'+args.model+'.json'), "w") as f:
            json.dump(judgement, f)
        # dump metric file
        with open(os.path.join(base_eval_path, localtime, 'metrics_check_'+args.model+'.json'), "w") as f:
            json.dump(metrics, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-4 evaluation on LVLMs.")
    parser.add_argument("--model", type=str, default='minigpt4', help="model")
    parser.add_argument("--gpu-id", type=str, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")


    parser.add_argument("--beam", type=int,default=5)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)

    parser.add_argument("--vg-path", type=str, default="/your/path/to/VG", help="path to vg file.")
    parser.add_argument("--shr-path", type=str, default="/your/path/to/shr", help="path to SHR annotation file.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repitition.")


    args = parser.parse_known_args()[0]




    # # Model
    # disable_torch_init()
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

    clip_scorer = CLIPModel(device=device)
    main(args)

