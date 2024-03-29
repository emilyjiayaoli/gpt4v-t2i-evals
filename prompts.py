'''
Prompt templates for Winogrande evaluation
@Author: Emily Li
'''

def VIEScore_baseline_prompt():
    system_msg = '''You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on the given rules. You will have to give your output in this way (Keep your reasoning concise and short.): \
                        {
                            \"score\" : [...],
                            \"reasoning\" : \"...\"
                        }'''
    user_msg = "RULES: The image is an AI-generated image according to the text prompt. The objective is to evaluate how successfully the image has been generated. On a scale 0 to 10: A score from 0 to 10 will be given based on the success in following the prompt. (0 indicates that the AI-generated image does not follow the prompt at all. 10 indicates the AI-generated image follows the prompt perfectly.) Put the score in a list such that output score = [score]. Text Prompt: {img_caption}"    
    # user_msg += "Provide your evaluation in JSON format, including keys for 'score' and 'reasoning'."
    return system_msg, user_msg


def GPT4_Eval_baseline_prompt():
    system_msg = ""
    user_msg = "Your task is to evaluate whether a given text caption accurately represents the main content and objects of an associated image. While the caption need not describe every detail of the image, it should convey the overall theme or subject. After your evaluation, rate the quality of the text caption's match to the image on a scale of 1-100, with 100 being a perfect match. Caption: '{img_caption}' "
    user_msg += "Provide your evaluation in JSON format, including keys for 'score' and 'reasoning'."
    return system_msg, user_msg


def get_raw_prob_num_prompts(output_json=True):
    '''Describe the image is removed '''
    system_msg = ""
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely."
    user_msg = "Calculate a similarity score in the format of the probability of a match between the provided image and the caption '{img_caption}'. Explain the rationale \n" 
    if output_json:
        user_msg += "Provide your evaluation in JSON format, including keys for 'score', and 'reasoning'. "
    return system_msg, user_msg


def get_prob_num_prompts():
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely."
    user_msg = "First describe what is happening in the image. Then, calculate a similarity score in the format of the probability of a match between the provided image and the caption '{img_caption}'. Explain the rationale \n" 
    user_msg += "Provide your evaluation in JSON format, including keys for 'score', 'description, and 'reasoning'."
    return system_msg, user_msg
    
def get_binary_scoring_prompts():
    '''
    Yes-or-no ("Does the caption correctly describe this image?")
    '''
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely."

    user_msg = "Is '{img_caption}' the correct description for the image? Please answer yes or no."
    user_msg += "Provide your evaluation in JSON format, including keys for 'score': 'yes'/'no' and 'reasoning'."
    return system_msg, user_msg

def get_likert_scale_prompts(prompt_type:str='agreement'):
    '''
    Classify the "very similar", "similar", "different"
    '''
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely."
    if prompt_type == 'likelihood':
        user_msg = """Assess the likelihood that the provided image matches the caption ‘{img_caption}’. Rate the likelihood of match using the following scale.
            1. 'Very Likely', 
            2. 'Somewhat Likely', 
            3. 'Neutral', 
            4. 'Somewhat Unlikely', 
            5. 'Very Unlikely'
            Explain your rationale."""
    elif prompt_type == 'similarity':
        user_msg = """Compare and contrast the provided image with the caption ‘{img_caption}’. How similar are they? Rate their similarity using the following scale.
            1. 'Very Similar', 
            2. 'Somewhat Similar', 
            3. 'Neutral', 
            4. 'Somewhat Different', 
            5. 'Very Different'
            Explain your rationale."""
    elif prompt_type == 'agreement':
        user_msg = """Assess how well the provided image matches the caption ‘{img_caption}’. Rate your agreement with the match using the following scale.
            1. 'Perfect Agreement', 
            2. 'High Agreement', 
            3. 'Moderate Agreement', 
            4. 'Low Agreement', 
            5. 'No Agreement'
            Explain your rationale."""
    
    
    user_msg += "\nProvide your evaluation in JSON format, including keys for 'score': '<Category>' and 'reasoning'."

    return system_msg, user_msg

def get_num_score_prompts():
    '''Describe the image is removed '''
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely. "
    user_msg = "Calculate a similarity score between 0 and 100, where 0 indicates no match and 100 indicates perfect match between the provided image and the caption '{img_caption}'. Explain your rationale \n" 
    user_msg += "Provide your evaluation in JSON format, including keys for 'score', and 'reasoning'."
    return system_msg, user_msg

def get_better_caption_prompt():
    ''' Only compatible with 2 image captions {img_caption1} and {img_caption2}'''
    system_msg = "You are a helpful image data analyst and linguist. Answer the user’s questions accurately and concisely. "
    user_msg = "Which of the folowing caption is a better match for the provided image? Caption 1: '{img_caption1}'. Caption 2: '{img_caption2}'. Explain your rationale \n" 
    user_msg += "Provide your evaluation in JSON format, including keys for 'score': 'Caption 1'/'Caption 2', and 'reasoning'."
    return system_msg, user_msg