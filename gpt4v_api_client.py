'''
GPT4V API Client
@Author: Emily Li
'''

import json
import time
import traceback
import requests 
import os

class GPT4V_API_Client:
    def __init__(self, log_folder_path, openai_api_key, 
                 post_processing_fn=None, 
                 max_tokens=1200, max_retries=1, delay_time=10, 
                 score_none_retry_msg=""):
        self.max_tokens = max_tokens  # return max 1200 tokens per call
        self.max_retries = max_retries
        self.delay_time = delay_time # seconds to delay

        # Global variable to keep track of tokens used in the last minute
        self.tokens_used_last_minute = 0
        self.last_reset_time = time.time()

        self.openai_api_key = openai_api_key

        self.reset_api_key(openai_api_key)
        self.log_folder_path = log_folder_path

        self.score_none_retry_msg = score_none_retry_msg

        self.post_processing_fn = post_processing_fn

    def reset_api_key(self, openai_api_key:str):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
            }
    
    def log_to_file(self, folder_name, file_name, content): # Function to create a log file and write logs
        folder_path = os.path.join(folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'a') as file:
            file.write(content + '\n')

    def is_error_response(self, response): # helper
        return 'error' in response

    def handle_error_response(self, response, retry_count): # helper
        error_type = response['error']['type']

        if retry_count >= self.max_retries:
            self.log_to_file(self.log_folder_path, 'api_errors.txt', 'Max retries exceeded')
            print("Already hit max_retries", response['error'])
            return None, None

        if error_type == 'server_error':
            # Retry the call immediately
            print("Server error. Retrying...")
            return self.process_gpt4_v_response(self.call_gpt4v(self.image, self.system_msg, self.user_msg), retry_count + 1)

        elif error_type == 'requests' and response['error']['code'] == 'rate_limit_exceeded': # RPD > 100/day --> switch keys
            print("RPD limit reached, need to switch keys...")
            assert('RPD' in response['error']['message'])
            return None, None
        
        elif error_type in ['tokens', 'invalid_request_error']:
            # Delay and then retry the call
            print(f"TPM limit reached, delaying by {self.delay_time}, then retrying...")
            time.sleep(self.delay_time)  # Delay for delay_time seconds, adjust as needed
            return self.process_gpt4_v_response(self.call_gpt4v(self.image, self.system_msg, self.user_msg), retry_count + 1)
        
        else:
            # Unhandled error
            print("Unhandled error", response)
            return None, None
        
    def process_response_content(self, response): # helper
        # process valid response
        content = response['choices'][0]['message']['content'] # Get content field from the response
        # find ```json\n{\n and get everything after starting from the first char
        content = content[content.find('{'):]

        if self.post_processing_fn:
            # organize raw output into json format
            json_data = self.post_processing_fn(content)
        else:
            content_cleaned = content.replace('```json\n', '').replace('\n```', '') # Remove the Markdown code block formatting (the triple backticks and 'json' line)
            json_data = json.loads(content_cleaned) # Parse with json.loads()
        tokens_used = response['usage']['total_tokens']

        return json_data, tokens_used

    # Main processing
    def process_gpt4_v_response(self, response, retry_count=0):
        try:
            # Handle errors if any
            if self.is_error_response(response):
                return self.handle_error_response(response, retry_count)

            data_content, tokens_used = self.process_response_content(response)

            if "score" not in data_content or not tokens_used:
                # raise Exception("Error in process_response_content")
                # if data_content['score'] is None:
                print("'score' not in data_content, retrying...")
                self.process_gpt4_v_response(self.call_gpt4v(self.image, self.system_msg, self.user_msg), retry_count + 1)
            
            # Process response
            return data_content, tokens_used

        except Exception as e:
            error_message = traceback.format_exc()
            self.log_to_file(self.log_folder_path, 'api_errors.txt', error_message)
            print("Error occured in process_gpt4_v_response", e) #TODO:delete
            print("Response:", response)
            traceback.print_exc()
            return None, None
    

    # Function that calls GPT4v API given an image and prompt
    def call_gpt4v(self, base64_image, system_msg, user_msg):
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": [{
                                "type": "text",
                                "text": user_msg
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
            }
            ],
            "max_tokens": self.max_tokens
        }
        
        # Saves image, prompt incase re-call needed
        self.image = base64_image
        self.system_msg = system_msg
        self.user_msg = user_msg

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)

        try:
            self.log_to_file(self.log_folder_path, 'api_raw_output.txt', str(response.json())) # Log the raw API response
        except Exception as e:
            print("Error in logging raw output", e)
            print("Response:", response)
            traceback.print_exc()
        
        return response.json()
    
    
