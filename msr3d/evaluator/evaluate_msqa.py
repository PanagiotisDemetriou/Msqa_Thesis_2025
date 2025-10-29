from utils import load_json, execute_chat, EM_Evaluator
import re
import os
import torch
from tqdm import tqdm
from types import SimpleNamespace

def extract_question(text):
    # Using regular expression to find text between 'USER:' and 'ASSISTANT:'
    match = re.search(r"USER: (.*?) ASSISTANT:", text)
    return match.group(1) if match else None

def extract_number(text):
    # Using regular expression to find number in text
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None

class LLMEvaluator():
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"total_cnt": 0}
        self.metric_type_list = ['gpt_score', 'em1', 'em1_strict', 'cider', 'bleu', 'meteor', 'rouge']
        for metric_type in self.metric_type_list:
            self.eval_dict[metric_type] = 0
    
    def update(self, score_dict):
        '''
            update the evaluation results
        '''
        for metric_type in score_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] += score_dict[metric_type]
        self.eval_dict["total_cnt"] += 1
    
    def summary(self):
        '''
        '''
        # self.eval_dict["gpt_score"] = self.eval_dict["gpt_score"]/self.eval_dict["total_cnt"]
        for metric_type in self.eval_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] = self.eval_dict[metric_type]/self.eval_dict["total_cnt"]
        return self.eval_dict

    def get_gpt_score(self, question, answer, gt):
        '''
            evaluate the results
        '''
        model = self.cfg.gpt_model # from configs  "gpt-4o-2024-08-06"
        api_key = self.cfg.api_key # from configs its empty
        api_version = self.cfg.api_version # from configs "azure"
        region = self.cfg.region # from configs its empty
        messages = load_json(self.cfg.gpt_score_prompt_path) 
        user_prompt = "\n".join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({"role": "user", "content": user_prompt})
        response = execute_chat(messages, api_version, api_key, model, region) # get response from gpt
        score = extract_number(response) # extract score from response
        return score

class MSQAEvaluator():
    '''
        process the data from files
    '''
    def __init__(self, config):
        self.cfg = SimpleNamespace(**config)
        self.eval_dict = {"gpt_score": 0, "total_cnt": 0}
        self.evaluator = LLMEvaluator(self.cfg)

    def eval_metrics(self):
        
        dataset_names_list = self.cfg.evaluate_dataset # only scannet 
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score' # turned off - gpt evaluates each metic and gives a score 0-100
        result_dict = load_json(self.cfg.result_file) # json containing the predictions
        result_scores_dict = {} # to store the scores for each dataset
        for dataset_name in dataset_names_list:
            result_dict_list = result_dict[dataset_name] # list of dicts for each dataset from json 
            score_list = [] # to store scores for each instance
            for i in tqdm(range(len(result_dict_list))):
                result_dict = result_dict_list[i] # load the a dict from the list
                if 'question' in result_dict: 
                    question = result_dict["question"] # get the question
                else:
                    if "instruction" in result_dict:
                        question = extract_question(result_dict["instruction"])  
                gt = result_dict["response_gt"][0] # get the ground truth answer
                answer = result_dict["response_pred"] # get the predicted answer
                index = result_dict["index"] # get the index
                scored_dict = {"question": question, "answer": answer, "gt": gt} # make a dict from extracted info of json 
                if self.cfg.gpt_score_flag:# if gpt score flag is true (turned off)
                    score = self.evaluator.get_gpt_score(question, answer, gt)
                    scored_dict['gpt_score'] = (score-1)*25 # scale to 0-100 (score is between 1-5)
                lang_evaluator = EM_Evaluator() # evaluator for exact and partial matches
                scored_dict.update(lang_evaluator.eval_instance(answer, [gt])) # get the scores for em1 and em1_strict
                self.evaluator.update(scored_dict) # update the evaluator with the scores
                if 'type' in result_dict:
                    scored_dict['type'] = result_dict['type'] # add the type of question if exists
                score_list.append(scored_dict) # append the scored dict to score list
            result_scores_dict[dataset_name] = score_list # store the score list for the dataset
    
        QA_type_list = [
            "counting",
            "existence",
            "attribute",
            "spatial relationship",
            "navigation",
            "refer",
            "affordance",
            "description",
            "room type",
        ]
        statistic_dict = {'scannet': {}, 
                         'RScan': {},
                         'ARKitScenes': {}}
        
        metric_type_list = ['em1', 'em1_strict']
        result_dict = {}
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        if file_tag == 'with_gpt_score':
            metric_type_list.append('gpt_score')

        for dataset_name in dataset_names_list:
            scores_data = result_scores_dict[dataset_name]
            data_instance_cnt = 0
            for data_instance in scores_data:
                if 'type' in data_instance:
                    data_QA_type = data_instance['type'] # get the type of each question
                else:
                    ValueError("No type in data_instance")
                for metric_type in metric_type_list:
                    if metric_type not in statistic_dict[dataset_name]:
                        statistic_dict[dataset_name][metric_type] = {}
                    for QA_type in QA_type_list:
                        if QA_type in data_QA_type.lower():
                            if QA_type not in statistic_dict[dataset_name][metric_type]:
                                statistic_dict[dataset_name][metric_type][QA_type] = {'score': [], 'cnt': 0, "avg": 0} 
                            statistic_dict[dataset_name][metric_type][QA_type]['score'].append(data_instance[metric_type]) # append score
                            statistic_dict[dataset_name][metric_type][QA_type]['cnt'] += 1 # increment count

        for dataset_name in dataset_names_list:
            for metric_type in metric_type_list:
                for QA_type in QA_type_list:
                    if QA_type in statistic_dict[dataset_name][metric_type]:
                        statistic_dict[dataset_name][metric_type][QA_type]['avg'] = sum(statistic_dict[dataset_name][metric_type][QA_type]['score'])/statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                        # compute average score for each QA type
        for metric_type in metric_type_list:
            statistic_dict[metric_type] = {'overall': {}} # overall scores across datasets
            for QA_type in QA_type_list:
                score_list = []
                cnt = 0
                for dataset_name in dataset_names_list:
                    if QA_type in statistic_dict[dataset_name][metric_type]:
                        # append weighted average scores
                        score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                        # calculate total count for averaging
                        cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                if cnt > 0:
                    statistic_dict[metric_type]['overall'][QA_type] = sum(score_list)/cnt # compute overall average score

        merged_QA_type_list = ['counting', 'existence', 'attribute_description', 'spatial_refer', 'navigation', 'others']
        for metric_type in metric_type_list:
            statistic_dict[metric_type]['merged'] = {}

        for metric_type in metric_type_list:
            for QA_type in merged_QA_type_list:
                score_list = []
                cnt = 0
                # for each QA type in merged list find their average scores across datasets
                for dataset_name in dataset_names_list:
                    if QA_type in ['counting', 'existence', 'navigation']:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                    elif QA_type == 'attribute_description':
                        if 'attribute' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['attribute']['avg'] * statistic_dict[dataset_name][metric_type]['attribute']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['attribute']['cnt']
                        if 'description' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['description']['avg'] * statistic_dict[dataset_name][metric_type]['description']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['description']['cnt']
                    elif QA_type == 'spatial_refer':
                        if 'spatial relationship' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['spatial relationship']['avg'] * statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt']
                        if 'refer' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['refer']['avg'] * statistic_dict[dataset_name][metric_type]['refer']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['refer']['cnt']
                    elif QA_type == 'others':
                        if 'affordance' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['affordance']['avg'] * statistic_dict[dataset_name][metric_type]['affordance']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['affordance']['cnt']
                        if 'room type' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['room type']['avg'] * statistic_dict[dataset_name][metric_type]['room type']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['room type']['cnt']
                    else:
                        ValueError("Invalid QA type")
                if cnt > 0:
                    statistic_dict[metric_type]['merged'][QA_type] = sum(score_list)/cnt
                    statistic_dict[metric_type]['merged'][QA_type + '_cnt'] = cnt
            statistic_dict[metric_type]['merged']['weighted_avg_score'] = sum([statistic_dict[metric_type]['merged'][QA_type] * statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])/sum([statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])
            result_dict = {}
            for key in statistic_dict['em1']['merged']:
                if 'cnt' in key:
                    continue
                if 'weighted' in key:
                    result_dict['EM-R_overall'] = statistic_dict['em1']['merged'][key]
                else:
                    result_dict[f'EM-R_{key}'] = statistic_dict['em1']['merged'][key]
            if 'gpt_score' in statistic_dict:
                for key in statistic_dict['gpt_score']['merged']:
                    if 'cnt' in key:
                        continue
                    if 'weighted' in key:
                        result_dict['GPT-Score_overall'] = statistic_dict['gpt_score']['merged'][key]
                    else:
                        result_dict[f'GPT-Score_{key}'] = statistic_dict['gpt_score']['merged'][key]
    
        return result_dict