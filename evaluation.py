'''
GPT-4V Evaluations
@Author: Emily Li
'''

import base64
import random
import os
import time
import numpy as np
import traceback
import ast
import json
from io import BytesIO

from gpt4v_api_client import GPT4V_API_Client
from dataset import get_dataset

class GPT4Vision_Evals:
    def __init__(self, save_folder_name:str, 
                 openai_api_key:str, 
                 dataset_name:str, 
                 root_dir:str='./', datasets_dir='./datasets',
                 post_processing_fn=None, 
                 system_prompt:str=None, user_prompt:str=None, 
                 api_max_retries=2, 
                 score_none_retry_msg="'score' MUST be a numerical value."):
        '''
        @param save_folder_name: name of folder to save results to
        @param openai_api_key: openai api key
        @param dataset_name: name of dataset to use
        @param root_dir: root directory of repo 
        @param post_processing_fn: function to post process the result score
        @param system_prompt: system prompt to use
        @param user_prompt: user prompt to use
        @param api_max_retries: max number of retries for api call
        @param score_none_retry_msg: message to print when score is None
        '''
        
        self.log_folder_path = os.path.join(root_dir, save_folder_name)

        # Define dataset
        print("Loading dataset from path...", dataset_name, end=" ")
        datasets_root_dir = os.path.join(root_dir, datasets_dir)
        self.dataset = get_dataset(root_dir=datasets_root_dir, dataset_name=dataset_name)
        print("Done!")
        
        self.system_msg = system_prompt
        self.user_msg = user_prompt

        self.post_processing_fn = post_processing_fn
        self.api = GPT4V_API_Client(self.log_folder_path, openai_api_key, post_processing_fn=self.post_processing_fn, max_retries=api_max_retries, score_none_retry_msg=score_none_retry_msg)

    def reset_api_key(self, openai_api_key:str):
        self.api.reset_api_key(openai_api_key=openai_api_key)

    # Function to encode the image
    def encode_image_from_path(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def encode_image_from_PIL(self, pil_image):
        # Convert the PIL Image to a byte stream
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG") 

        # Encode this image under base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Function to create a log file and write logs
    def log_to_file(self, folder_name, file_name, content):
        folder_path = os.path.join(folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'a') as file:
            file.write(content + '\n')

    def setup_prompt(self, img_caption):
        system_msg = self.system_msg
        user_msg = self.user_msg.replace('{img_caption}', img_caption)
        return system_msg, user_msg

    def post_process_score(self, raw_result_score):
        processed_result_score = None

        if isinstance(raw_result_score, str):

            if 'yes' in raw_result_score:
                processed_result_score = 1
                raw_result_score = 1
            elif 'no' in raw_result_score:
                processed_result_score = 0
                raw_result_score = 0

            elif 'Very Likely' in raw_result_score:
                processed_result_score = 1.0
            elif 'Somewhat Likely' in raw_result_score:
                processed_result_score = 0.75
            elif 'Neutral' in raw_result_score:
                processed_result_score = 0.5
            elif 'Somewhat Unlikely' in raw_result_score:
                processed_result_score = 0.25
            elif 'Very Unlikely' in raw_result_score:
                processed_result_score = 0.0

            elif 'Very Similar' in raw_result_score:
                processed_result_score = 1.0
            elif 'Somewhat Similar' in raw_result_score:
                processed_result_score = 0.75
            elif 'Neutral' in raw_result_score:
                processed_result_score = 0.5
            elif 'Somewhat Different' in raw_result_score:
                processed_result_score = 0.25
            elif 'Very Different' in raw_result_score:
                processed_result_score = 0.0
            
            
            elif 'Perfect Agreement' in raw_result_score:
                processed_result_score = 1.0
            elif 'High Agreement' in raw_result_score:
                processed_result_score = 0.75
            elif 'Moderate Agreement' in raw_result_score:
                processed_result_score = 0.5
            elif 'Low Agreement' in raw_result_score:
                processed_result_score = 0.25
            elif 'No Agreement' in raw_result_score:
                processed_result_score = 0.0

            # try:
            processed_result_score = float(raw_result_score)
            # except:
            #     # parse list of floats or ints using regex
            #     raw_result_score = re.findall(r"[-+]?\d*\.\d+|\d+", raw_result_score)
            #     try:
            #         processed_result_score = float(raw_result_score[0])
            #     except:
            #         print("Error in post_process_score:", raw_result_score)
            #         raise Exception("Error in post_process_score")

        elif isinstance(raw_result_score, list):
            if len(raw_result_score) > 1:
                print("len of raw_result_score > 1. raw_result_score:", raw_result_score)
                raise Exception("Error in post_process_score")
            processed_result_score = float(raw_result_score[0])
        else:
            processed_result_score = raw_result_score

        return processed_result_score

    ''' Given image path and a caption, returns a dictionary with keys 'score', 'description', and 'reasoning',
        e.g {'score': 95,
        'description': 'In the image, a large tree...',
        'reasoning': '...'}
    '''
    def evaluate_11match(self, image, img_caption:str) -> dict:

        system_msg, user_msg = self.setup_prompt(img_caption) # retrieve prompt

        if self.dataset.__class__.__name__ == 'EqBen_Mini':
            base64_image = self.encode_image_from_PIL(image)
        else:
            base64_image = self.encode_image_from_path(image) # encode image compatible w/ gpt4v api

        response = self.api.call_gpt4v(base64_image, system_msg, user_msg) 
        result, tokens = self.api.process_gpt4_v_response(response, retry_count=0)
        
        result['score'] = self.post_process_score(result['score'])

        return result, tokens
    
    def update_json_results(self, json_folder_path, json_filename, new_entries):
        ''' Reads existing_data = {
                0: {'images': 1...},
                1: {'images': 1...},
            }
            new_entries = {
                1: {'images': 2...},
                2: {'images': 3...},
            }
            updated_data = {
                0: {'images': 1...},
                1: {'images': 2...},
                2: {'images': 3...},
            }
        '''
        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path)
        file_path = os.path.join(json_folder_path, json_filename)
        
        try:
            # Load existing JSON object from the file
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, initialize with an empty dictionary
            existing_data = {}

        # Add new entries to the existing data
        for key, value in new_entries.items():
            existing_data[key] = value

        # Sort existing_data by key
        existing_data = dict(sorted(existing_data.items(), key=lambda item: int(item[0])))
        
        # Save the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def evaluate_gpt4v(self, sample_indices:list=None) -> dict:

        print("Saving to folder:", self.log_folder_path)

        failed_samples = []
        results = {}

        if not sample_indices:
            sample_indices = range(0, len(self.dataset))

        # Log prompt used, dataset name, and date and time the experiment was run
        log_content = "Date: " + time.strftime("%m/%d/%Y") + "\n"
        log_content += "Prompt:\n\t system_msg: " + self.system_msg + "\n"
        log_content += "\t user_msg: " + self.user_msg + "\n"
        log_content += "Dataset: " + self.dataset.__class__.__name__ + f". Number of samples: {len(self.dataset)}.\n\n"
        self.log_to_file(self.log_folder_path, 'info_log.txt', log_content)

        for i, id in enumerate(sample_indices): # Loop through each sample in the winoground dataset
            sample = self.dataset[id] # get sample
            sample_result = {}
            scores = np.zeros((len(sample['images']), len(sample['texts'])))

            try:
                print(f"Sample {id}. Finished...", end=" ")
                start_time = time.time()

                # get all combinations of image and caption
                for a, img in enumerate(sample['images']):
                    for b, caption in enumerate(sample['texts']):
                        result, token = self.evaluate_11match(img, caption)
                        if result['score'] is None or not token:
                            print(token, result)
                            raise Exception("Error in result")
                        
                        scores[a][b] = float(result['score'])
                        sample_result[f"i{a}c{b}_result"] = result
                        print(f"i{a}c{b} score:", result['score'], end=" ")

                # Populate sample result
                sample_result['scores'] = scores.tolist()

                if isinstance(sample['images'][0], str):
                    sample_result['images'] = sample['images']
                sample_result['texts'] = sample['texts']

                print(f"Took {round(time.time() - start_time, 3)} seconds!")

                results[str(id)] = sample_result # add sample result to results dictionary

                if id % 10 == 0: # save results every 10 samples
                    self.update_json_results(self.log_folder_path, 'score_results_log.json', results)
                    results = {}

            except Exception as e:
                print(sample)
                print(f"Error in sample {id}: {e}. Saving results so far. Then returning...", end=" ")
                self.update_json_results(self.log_folder_path, 'score_results_log.json', results)
                results = {}
                failed_samples.append(id)
                self.log_to_file(self.log_folder_path, 'info_log.txt', 'Failed Examples: [' + ', '.join([str(i) for i in failed_samples]) + ']')
                traceback.print_exc()
                return
        
        print(f"Done! Saving results to {self.log_folder_path}score_results_log.json")
        self.update_json_results(self.log_folder_path, 'score_results_log.json', results)
        print("Failed samples:", failed_samples)
        self.log_to_file(self.log_folder_path, 'info_log.txt', 'Failed Examples: [' + ', '.join([str(i) for i in failed_samples]) + ']')


    def get_scores(self, json_result_path:str):
        ''' Get score matrix from results json file with output shape (len(dataset), # images, # captions)'''
        with open(json_result_path, 'r') as file:
            data = json.load(file)
            assert len(data) >= 1

            b, h, w = len(self.dataset), len(list(data.values())[0]['scores']), len(list(data.values())[0]['scores'][0])
            scores = np.zeros((b, h, w))

            for sample_id in data.keys():
                scores[int(sample_id)] = data[sample_id]['scores']

            return scores
        
    def get_results(self, json_result_path='score_results_log.json', get_overall_scores_matrix=False):
        overall_scores = self.get_scores(os.path.join(self.log_folder_path, json_result_path))
        if get_overall_scores_matrix:
            return overall_scores
        else:
            return self.dataset.evaluate_scores(overall_scores)

    # Read the file path and process the file
    def get_results_outdated(name: str, file_path: str, return_entirety=False):
        scores = {}
        ids = []
        total_image_score = 0
        total_text_score = 0
        total_group_score = 0
        total_samples = 0

        if return_entirety:
            all_data = {}

        # Store indices failure cases
        failure_cases = {'text_score_fc': [],
                        'image_score_fc': [],
                        'group_score_fc': [],
                        'total_ids': []}

        with open(file_path, 'r') as file:
            while True:
                # Read two lines from the file
                line1 = file.readline()
                line2 = file.readline()

                # Check if either line is empty (end of file)
                if not line1 or not line2:
                    break

                line = (line1.strip() + line2.strip()).replace("array(", "").replace("])", "]").replace(".,", ",").replace(".]", "]")
                data = ast.literal_eval(line)

                # Extract the ID
                if data['id'] in ids:
                    print(f"Dup found! ID {data['id']}. skipping...")
                    continue

                else:
                    id = data['id']

                if return_entirety:
                    all_data[id] = data

                ids.append(id)

                # Extract and count scores
                image_score = data['image_score']
                text_score = data['text_score']
                group_score = data['group_score']

                # Append the results
                scores[id] = {'image_score': image_score, 'text_score': text_score, 'group_score': group_score}
                total_image_score += image_score
                total_text_score += text_score
                total_group_score += group_score
                total_samples += 1

                if image_score == 0: failure_cases['image_score_fc'].append(id)
                if text_score == 0: failure_cases['text_score_fc'].append(id)
                if group_score == 0: failure_cases['group_score_fc'].append(id)
                failure_cases['total_ids'].append(id)


        print("Experiment:", name, "- Total samples:", total_samples)
        print(" Image Score:", total_image_score, "-->", round(total_image_score/total_samples, 3))
        print(" Text Score:", total_text_score, "-->", round(total_text_score/total_samples, 3))
        print(" Group Score:", total_group_score, "-->", round(total_group_score/total_samples, 3))
        
        if return_entirety:
            return sorted(ids), scores, failure_cases, all_data
        else:
            return sorted(ids), scores, failure_cases
    

    ''' 
    @brief Returns a list of randomly sampled indices from Winoground dataset
    Default: sample 200 from 400 or half of the dataset
    '''
    def randomly_sample(amt_to_sample:int=200, indices_len:int=400, seed=11182023):
        assert(amt_to_sample <= indices_len)
        random.seed(seed)

        indices = range(0, indices_len)
        samples = random.sample(indices, amt_to_sample) # shuffle the list & sample the specified amt
        samples.sort()
        return samples
