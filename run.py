from evaluation import GPT4Vision_Evals as evals

# 1) Define prompt or import existing prompts from prompt.py
def VIEScore_baseline_prompt():
    system_msg = '''You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on the given rules. You will have to give your output in this way (Keep your reasoning concise and short.): \
                        {
                            \"score\" : [...],
                            \"reasoning\" : \"...\"
                        }'''
    user_msg = "RULES: The image is an AI-generated image according to the text prompt. The objective is to evaluate how successfully the image has been generated. On a scale 0 to 10: A score from 0 to 10 will be given based on the success in following the prompt. (0 indicates that the AI-generated image does not follow the prompt at all. 10 indicates the AI-generated image follows the prompt perfectly.) Put the score in a list such that output score = [score]. Text Prompt: {img_caption}"    
    # user_msg += "Provide your evaluation in JSON format, including keys for 'score' and 'reasoning'."
    return system_msg, user_msg

# 2) Define other args
system_msg, user_msg = VIEScore_baseline_prompt()               # Defines prompt (change me)
dataset_name = "TIFA160_DSG"                                    # Dataset name (change me) - see dataset.py for all provided datasets
eval_method_name = "GPT4V-Eval"                                   # Method name (change me)
save_folder_name = f"logs/{dataset_name}/{eval_method_name}/"   # Log folder path (change me)
openai_api_key = 'YOUR API KEY'

# 3) Define evaluator & Run evaluation
evaluator = evals(save_folder_name, openai_api_key, dataset_name=dataset_name,
                  post_processing_fn=None, # Post processing function to apply to the generated text. Default: None
                  system_prompt=system_msg, user_prompt=user_msg, # Prompt to use for evaluation
                  api_max_retries=2, # Number of times to retry API call before giving up
                  )

# 4) Reset API key (optional)
evaluator.reset_api_key(openai_api_key) # Ensure that the correct API key is used

# 5) Run evaluation  
# evaluator.evaluate_gpt4v(list(range(0,2)))    # Option 1: Evaluate by list of ids
evaluator.evaluate_gpt4v()                     # Option 2: Evaluate all ids in dataset

# 6) Get results: Results are automatically saved every 10 samples or when error occurs, and will be loaded automatically when evaluator is re-initialized
results = evaluator.get_results()
