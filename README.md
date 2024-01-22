# GPT4v Evaluator for text-image alignment scoring
## What this is about
Evaluating api-based multimodal Vision Language Models (VLMS) e.g GPT4v, Imagen2 as an evaluator of text-image alignment. In simple words, we evaluate such VLMs' abilities to output a a numerical score indicating the similarity/alignment between one image and one text. Think about using GPT4v as CLIPScore. This repo supports generalization to calculating an m x n matrix of alignment scores between m images and n texts. We provide support for evaluations on over 5 popular benchmarks evaluating visio-linguistic compositionalities of such scoring methods.


## Installation
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Datasets
### Supported datasets

```     Winoground,
        EqBen_Mini,
        TIFA160_DSG,
        Flickr8K_Expert,
        Flickr8K_CF,
```

### Quickstart
Run run.py
```python
from evaluation import GPT4Vision_Evals as evals
from prompts import VIEScore_baseline_prompt # Or define your own, exmaples in prompts.py

# 2) Define args
system_msg, user_msg = VIEScore_baseline_prompt()               # Defines prompt (change me)
dataset_name = "TIFA160_DSG"                                    # Dataset name (change me) - see dataset.py for all provided datasets
eval_method_name = "GPT4V-Eval"                                   # Method name (change me)
save_folder_name = f"sample-logs/{dataset_name}/{eval_method_name}/"   # Log folder path (change me)
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
```


### Add your own dataset
Fill in and append the following class template to `dataset.py`. Please view Dataset class for supported datasets as an example.
```python
class CustomDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx) -> dict:
        # @return: dict 
            # example: {
            #     "images": [image_0, image_1],
            #     "texts": [caption_0, caption_1]
            # }
        pass

    def evaluate_scores(self, scores):
        # @param: scores - a multi-dim list score matrix of shape (len(CustomDataset), # images per sample, # texts per sample)
        pass
```
