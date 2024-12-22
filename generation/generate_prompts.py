import openai
import json
import logging
import argparse
import os

logging.basicConfig(filename='gpt_generate.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def gen_sentence_gpt(sc, gen_prompt, openai.api_key):
    """
    Generate a sentence using GPT-3.5-turbo model.

    Args:
        sc (str): The subject category to be included in the prompt.
        gen_prompt (str): The prompt template with a placeholder for the subject category.

    Returns:
        str: The generated sentence from the GPT model.
    """
    gen_prompt = gen_prompt.replace("{sc}", sc)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": gen_prompt}], temperature=0.7, max_tokens=4000)
    print(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]

def gen_sentence_gpt_4(sc, gen_prompt, openai.api_key):
    """
    Generate a sentence using GPT-4-0613 model.

    Args:
        sc (str): The subject category to be included in the prompt.
        gen_prompt (str): The prompt template with a placeholder for the subject category.

    Returns:
        str: The generated sentence from the GPT model.
    """
    gen_prompt = gen_prompt.replace("{sc}", sc)
    response = openai.ChatCompletion.create(model="gpt-4-0613", messages=[{"role": "user", "content": gen_prompt}], temperature=0.7, max_tokens=4000)
    print(response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]

def process_class_list(models, args):
    """
    Process the class list and generate prompts for each class.

    Args:
        models (dict): Dictionary of model functions.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary of prompts for each class.
    """
    prompts_dict = {}
    class_json_file = os.path.join(args.class_info_path, f"{args.dataset}_classes_{args.class_subset}.json")

    with open(class_json_file, 'r') as f:
        classes = json.load(f)
        for class_token, class_name in classes["class_to_name"].items():
            prompts = models[args.prompt_model](class_name, args.gen_prompt)
            prompt_list = [prompt.split('. ')[1] for prompt in prompts.split('\n')]
            logging.info(class_name)
            logging.info(prompts)
            logging.info("Received " + str(len(prompt_list)) + " prompts.")
            if len(prompt_list) != 100:
                logging.info("INCORRECT PROMPTS GENERATED")
            prompts_dict[class_token] = prompt_list
    return prompts_dict

def save_to_json(prompts_dict, output_json_file):
    """
    Save the prompts dictionary to a JSON file.

    Args:
        prompts_dict (dict): Dictionary of prompts for each class.
        output_json_file (str): Path to the output JSON file.
    """
    with open(output_json_file, 'w') as json_file:
        json.dump(prompts_dict, json_file, indent=2)
