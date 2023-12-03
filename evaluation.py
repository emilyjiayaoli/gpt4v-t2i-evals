'''
Winoground Evals on GPT-4V
@Author: Emily Li
'''

import base64
import random
import os
import time
import winoground
import numpy as np
import traceback
import ast

from gpt4v_api_client import GPT4V_API_Client

class GPT4V_Winoground_Evals:
    def __init__(self, log_path:str, save_folder_name:str, 
                 openai_api_key:str, post_processing_fn=None, 
                 system_prompt:str=None, user_prompt:str=None, 
                 api_max_retries=1, dataset_root_dir='./', 
                 score_none_retry_msg="'score' MUST be a numerical value."):
        # self.token_per_min_limit = 8000
        # self.max_tokens = 1200  # return max 1200 tokens per call
        self.log_folder_path = os.path.join(log_path, save_folder_name)

        self.dataset = winoground.Winoground(root_dir=dataset_root_dir, return_image_paths=True)

        self.system_msg = system_prompt
        self.user_msg = user_prompt

        self.winoground_scores = None
        self.accuracy = None
        self.winoground_analysis = None

        self.post_processing_fn = post_processing_fn
        self.api = GPT4V_API_Client(self.log_folder_path, openai_api_key, post_processing_fn=self.post_processing_fn, max_retries=api_max_retries, score_none_retry_msg=score_none_retry_msg)

    def reset_api_key(self, openai_api_key:str):
        self.api.reset_api_key(openai_api_key=openai_api_key)

    # Function to update the global token count and possibly delay
    def update_token_usage_and_delay(self, tokens_used):
        current_time = time.time()
        time_since_last_reset = current_time - self.last_reset_time
        
        # Reset token count every minute
        if time_since_last_reset >= 60:
            self.tokens_used_last_minute = 0
            self.last_reset_time = current_time
        
        # Update the token count
        self.tokens_used_last_minute += tokens_used

        # If the token count exceeds the limit, delay until the rate is lower
        if self.tokens_used_last_minute > self.token_per_min_limit:
            time_to_wait = 60 - time_since_last_reset
            print(f"Rate limit exceeded: {self.tokens_used_last_minute} tokens used in the last minute. Delaying next calls for {time_to_wait}s.")
            time.sleep(time_to_wait)
            self.tokens_used_last_minute = 0
            self.last_reset_time = time.time()

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

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

            if raw_result_score == 'yes':
                processed_result_score = 1
            elif raw_result_score == 'no':
                processed_result_score = 0

            if 'Very Likely' in raw_result_score:
                processed_result_score = 1.0
            elif 'Somewhat Likely' in raw_result_score:
                processed_result_score = 0.75
            elif 'Neutral' in raw_result_score:
                processed_result_score = 0.5
            elif 'Somewhat Unlikely' in raw_result_score:
                processed_result_score = 0.25
            elif 'Very Unlikely' in raw_result_score:
                processed_result_score = 0.0

            if 'Very Similar' in raw_result_score:
                processed_result_score = 1.0
            elif 'Somewhat Similar' in raw_result_score:
                processed_result_score = 0.75
            elif 'Neutral' in raw_result_score:
                processed_result_score = 0.5
            elif 'Somewhat Different' in raw_result_score:
                processed_result_score = 0.25
            elif 'Very Different' in raw_result_score:
                processed_result_score = 0.0
            
            
            if 'Perfect Agreement' in raw_result_score:
                processed_result_score = 1.0
            elif 'High Agreement' in raw_result_score:
                processed_result_score = 0.75
            elif 'Moderate Agreement' in raw_result_score:
                processed_result_score = 0.5
            elif 'Low Agreement' in raw_result_score:
                processed_result_score = 0.25
            elif 'No Agreement' in raw_result_score:
                processed_result_score = 0.0

            processed_result_score = float(raw_result_score)
            
        else:
            processed_result_score = raw_result_score

        return processed_result_score

    ''' Given image path and a caption, returns a dictionary with keys 'score', 'description', and 'reasoning',
        e.g {'score': 95,
        'description': 'In the image, a large tree...',
        'reasoning': '...'}
    '''
    def evaluate_11match(self, image_path:str, img_caption:str) -> dict:

        system_msg, user_msg = self.setup_prompt(img_caption) # retrieve prompt

        base64_image = self.encode_image(image_path) # encode image compatible w/ gpt4v api
        
        response = self.api.call_gpt4v(base64_image, system_msg, user_msg) 
        result, tokens = self.api.process_gpt4_v_response(response, retry_count=0)

        # if result['score'] is None:
        #     print("Score is None, retrying in eval11...")
        #     retry_user_msg = user_msg + " " + self.score_none_retry_msg
        #     response = self.api.call_gpt4v(base64_image, system_msg, retry_user_msg)
        #     result, tokens = self.api.process_gpt4_v_response(response, retry_count=0)
        
        result['score'] = self.post_process_score(result['score'])

        return result, tokens


    ''' Winoground eval metrics'''
    def text_correct(self, i1c1, i2c1, i1c2, i2c2) -> bool:
        return float(i1c1)*100 > float(i1c2)*100 and float(i2c2)*100 > float(i2c1)*100
    def image_correct(self, i1c1, i2c1, i1c2, i2c2) -> bool:
        return float(i1c1)*100 > float(i2c1)*100 and float(i2c2)*100 > float(i1c2)*100
    def group_correct(self, image_correct:bool, text_correct:bool) -> bool:
        return image_correct and text_correct

    def evaluate_winoground_gpt4v(self, sample_indices:list) -> dict:

        print("Saving to folder:", self.log_folder_path)

        failed_samples = []

        for i, id in enumerate(sample_indices): # Loop through each sample in the winoground dataset
            sample = self.dataset[id] # get sample
            sample_result = {}
            scores = np.zeros((2,2))

            try:
                # get image and caption
                img1_path, img2_path = sample['image_options']
                caption1, caption2 = sample['caption_options']
                
                print(f"Sample {sample['id']}...", end=" ")
                start_time = time.time()

                # get all combinations of image and caption
                i1c1_result, token1 = self.evaluate_11match(img1_path, caption1)
                if i1c1_result['score'] is None or not token1: 
                    print(token1, i1c1_result)
                    raise Exception("Error in i1c1_result")
                print("Finished i1c1,", end=" ")

                i1c2_result, token2 = self.evaluate_11match(img1_path, caption2)
                if i1c2_result['score'] is None or not token2: 
                    print(token2, i1c2_result)
                    raise Exception("Error in i1c2_result")
                print("i1c2,", end=" ")

                i2c1_result, token3 = self.evaluate_11match(img2_path, caption1)
                if i2c1_result['score'] is None or not token3: 
                    print(token3, i2c1_result)
                    raise  Exception("Error in i2c1_result")
                # self.update_token_usage_and_delay(token3)
                print("i2c1,", end=" ")

                i2c2_result, token4 = self.evaluate_11match(img2_path, caption2)
                if i2c2_result['score'] is None or not token4:
                    print(token4, i2c2_result)
                    raise Exception("Error in i2c2_result")
                # self.update_token_usage_and_delay(token4)
                print("i2c2,", end=" ")

                # Populate sample result
                sample_result['id'] = sample['id']

                scores[0][0] = float(i1c1_result['score'])
                scores[0][1] = float(i1c2_result['score'])
                scores[1][0] = float(i2c1_result['score'])
                scores[1][1] = float(i2c2_result['score'])
                sample_result['scores'] = scores

                # calculate image score, text score, and group score
                sample_result['text_score'] = int(self.text_correct(i1c1_result['score'], i2c1_result['score'], i1c2_result['score'], i2c2_result['score']))
                sample_result['image_score'] = int(self.image_correct(i1c1_result['score'], i2c1_result['score'], i1c2_result['score'], i2c2_result['score']))
                sample_result['group_score'] = int(self.group_correct(sample_result['text_score'], sample_result['image_score']))

                # save other info
                sample_result['img1_path'] = img1_path
                sample_result['img2_path'] = img2_path
                sample_result['caption1'] = caption1
                sample_result['caption2'] = caption2

                sample_result['i1c1_result'] = i1c1_result
                sample_result['i1c2_result'] = i1c2_result
                sample_result['i2c1_result'] = i2c1_result
                sample_result['i2c2_result'] = i2c2_result

                # log results
                log_content = f"Sample {sample['id']}, Group score: {sample_result['group_score']}, Time: {round(time.time() - start_time, 3)} seconds"
                self.log_to_file(self.log_folder_path, 'raw_evaluations_log.txt', log_content)

                # print sample summary
                print(f"Text score: {sample_result['text_score']},", end=" ")
                print(f"Image score: {sample_result['image_score']},", end=" ")
                print(f"Group score: {sample_result['group_score']},", end=" ")
                print(f"Took {round(time.time() - start_time, 3)} seconds!")

                self.log_to_file(self.log_folder_path, 'res_evaluations_log.txt', str(sample_result))

            except Exception as e:
                print(f"Error in sample {sample['id']}: {e}. Returning...", end=" ")
                failed_samples.append(sample['id'])
                self.log_to_file(self.log_folder_path, 'failed_samples.txt', '[' + ', '.join([str(i) for i in failed_samples]) + ']')
                traceback.print_exc()
                return
        
        print(f"Done - Overall failed_samples: {failed_samples}")
        self.log_to_file(self.log_folder_path, 'failed_samples.txt', '[' + ', '.join([str(i) for i in failed_samples]) + ']')

    # Read the file path and process the file
    def process_results(name: str, file_path: str):
        scores = {}
        ids = []
        total_image_score = 0
        total_text_score = 0
        total_group_score = 0
        total_samples = 0

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
