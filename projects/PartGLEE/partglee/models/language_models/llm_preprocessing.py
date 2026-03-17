# ------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Toyota Motor Europe NV/SA - INSAIT

# Toyota Motor Europe NV/SA, INSAIT, and their affiliates retain all intellectual property and proprietary rights in and to this software, 
# related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and related 
# documentation without an express license agreement from Toyota Motor Europe NV/SA and INSAIT is strictly prohibited.

# This file is part of the LangHOPS project.

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# See the LICENSE file in the repository root for details.
# ------------------------------------------------------------------------------------------------

import torch, re, random
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
SEG_TOKEN_INDEX = -201
CLS_TOKEN_INDEX = -202
REGION_TOKEN_INDEX = -203
REFER_TOKEN_INDEX = -204
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_TOKEN = "<seg>"
DEFAULT_CLS_TOKEN = "<cls>"
DEFAULT_REGION_TOKEN = "<region>"

GENERAL_CLASS_INDEX = -205
OBJ_CLASS_INDEX = -206
OBJ_PART_CLASS_INDEX = -207
OBJ_QUERY_INDEX = -208
OBJ_PART_QUERY_INDEX = -209
DEFAULT_GENERAL_CLS_TOKEN = "<cls>"
DEFAULT_OBJ_CLS_TOKEN = "<obj_cls>"
DEFAULT_PART_CLS_TOKEN = "<part_cls>"
DEFAULT_OBJ_QUERY_TOKEN = "<obj_query>"
DEFAULT_PART_QUERY_TOKEN = "<part_query>"

DEFAULT_GENERAL_CLS_IDENTIFIER = "<CLS>"
DEFAULT_GENERAL_CLS_IDENTIFIER_END = "</CLS>"
DEFAULT_OBJ_CLS_IDENTIFIER = "<OBJ>"
DEFAULT_PART_CLS_IDENTIFIER = "<OBJ-PART>"
DEFAULT_OBJ_QUERY_IDENTIFIER = "<OBJ-QUERY>"
DEFAULT_OBJ_QUERY_IDENTIFIER_END = "</OBJ-QUERY>"
DEFAULT_PART_QUERY_IDENTIFIER = "<OBJ-PART-QUERY>"
DEFAULT_PART_QUERY_IDENTIFIER_END = "</OBJ-PART-QUERY>"
ADDITIONAL_TOKENS = [DEFAULT_GENERAL_CLS_IDENTIFIER, DEFAULT_GENERAL_CLS_IDENTIFIER_END, DEFAULT_OBJ_CLS_IDENTIFIER, DEFAULT_PART_CLS_IDENTIFIER, DEFAULT_OBJ_QUERY_IDENTIFIER, DEFAULT_OBJ_QUERY_IDENTIFIER_END,  DEFAULT_PART_QUERY_IDENTIFIER, DEFAULT_PART_QUERY_IDENTIFIER_END]


def tokenizer_special_tokens(prompt, 
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX, 
        seg_token_index=SEG_TOKEN_INDEX, 
        cls_token_index=CLS_TOKEN_INDEX, 
        region_token_index=REGION_TOKEN_INDEX, 
        return_tensors=None):
    input_ids = []
    special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
    prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

    for chunk in prompt_chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        else:
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long).squeeze()
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    else:
        return input_ids

def preprocess_class_name(class_names, tokenizer, CLS_token='[CAT]'):

    tokenized = [tokenizer.encode(class_name, add_special_tokens=False) for class_name in class_names]
    tokenized_class_names = [tokens + [tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                tokenized]
    class_name_id = [token for sublist in tokenized_class_names for token in sublist]
    class_name_id = torch.tensor(class_name_id)
    cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
    cls_indices = torch.tensor(cls_indices)
    
    return class_name_id, cls_indices

def preencode_dataset_names(dataset_name_dict, tokenizer):
    # assume tokenizer has already tokens for ADDITIONAL_TOKENS
    for token in ADDITIONAL_TOKENS:
        id =  tokenizer.convert_tokens_to_ids(token)
        assert id != tokenizer.unk_token_id, f"tokenizer does not have token {token}"
    
    dataset_class_ids = {}
    dataset_class_ids_indices = {}
    
    for dataset_name in dataset_name_dict:
        class_names = dataset_name_dict[dataset_name]
        # tokenized = [tokenizer.encode(class_name, add_special_tokens=False) for class_name in class_names]
        # tokenized_class_names = [cls_identifier_token] + [  + tokens + [cls_identifier_token] for tokens in tokenized]
        # class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        # cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        # dataset_class_ids[dataset_name] = torch.tensor(class_name_id)
        # dataset_class_ids_indices[dataset_name] = torch.tensor(cls_indices)
        
        cls_identifier_token = tokenizer.encode(DEFAULT_GENERAL_CLS_IDENTIFIER, add_special_tokens=False)[0]
        cls_identifier_end_token = tokenizer.encode(DEFAULT_GENERAL_CLS_IDENTIFIER_END, add_special_tokens=False)[0]
        tokenized_class_names = [cls_identifier_token]
        class_indices = [0]
        for class_idx, class_name in enumerate(class_names):
            tokenized_class = tokenizer.encode(class_name, add_special_tokens=False)
            tokenized_class_names += tokenized_class
            class_indices += [class_idx+1] * len(tokenized_class)
            if class_idx != len(class_names)-1:
                tokenized_seperator = tokenizer.encode('/', add_special_tokens=False)
                tokenized_class_names += tokenized_seperator
                class_indices += [0] * len(tokenized_seperator)
        tokenized_class_names.append(cls_identifier_end_token)
        class_indices.append(0)
        dataset_class_ids[dataset_name] = torch.tensor(tokenized_class_names)
        dataset_class_ids_indices[dataset_name] = torch.tensor(class_indices)

    return dataset_class_ids, dataset_class_ids_indices
        
def tokenizer_conversation(sources, conv,  tokenizer, 
                           image_token=DEFAULT_IMAGE_TOKEN,
                           image_index=IMAGE_TOKEN_INDEX,
                           general_cls_token=DEFAULT_GENERAL_CLS_TOKEN,
                           general_cls_index=GENERAL_CLASS_INDEX,
                           obj_cls_token=DEFAULT_OBJ_CLS_TOKEN,
                           obj_cls_index=OBJ_CLASS_INDEX,
                           part_cls_token=DEFAULT_PART_CLS_TOKEN,
                           part_cls_index=OBJ_PART_CLASS_INDEX,
                           obj_query_token=DEFAULT_OBJ_QUERY_TOKEN,
                           obj_query_index=OBJ_QUERY_INDEX,
                           part_query_token=DEFAULT_PART_QUERY_TOKEN,
                           part_query_index=OBJ_PART_QUERY_INDEX, 
                           return_tensors=None):
    
    special_token_map = {
        image_token: image_index,
        general_cls_token: general_cls_index,
        obj_cls_token: obj_cls_index,
        part_cls_token: part_cls_index,
        obj_query_token: obj_query_index,
        part_query_token: part_query_index,
    }
    
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        
    split_texts = '(' + '|'.join(special_token_map.keys()) + ')'
    print("split_texts: ", split_texts)
    input_ids_batch = []
    for prompt in conversations:
        input_ids = []
        prompt_chunks = re.split(split_texts, prompt)
        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        input_ids =  torch.tensor(input_ids, dtype=torch.long).squeeze()
        input_ids_batch.append(input_ids)
    input_ids_batch = torch.stack(input_ids_batch, dim=0)
    
    return input_ids_batch

def generate_prompts_archive(
    num_obs, 
    num_parts, 
    general_cat_id = DEFAULT_GENERAL_CLS_IDENTIFIER,
    obj_cat_id = DEFAULT_OBJ_CLS_IDENTIFIER,
    part_cat_id = DEFAULT_PART_CLS_IDENTIFIER,
    obj_query_id = DEFAULT_OBJ_QUERY_IDENTIFIER,
    obj_query_id_end = DEFAULT_OBJ_QUERY_IDENTIFIER_END, 
    part_query_id = DEFAULT_PART_QUERY_IDENTIFIER,
    part_query_id_end = DEFAULT_PART_QUERY_IDENTIFIER_END, 
    image_token_ph=DEFAULT_IMAGE_TOKEN,
    general_category_ph=DEFAULT_GENERAL_CLS_TOKEN,
    obj_categores_ph=DEFAULT_OBJ_CLS_TOKEN,
    part_categores_ph=DEFAULT_PART_CLS_TOKEN,
    obj_query_ph=DEFAULT_OBJ_QUERY_TOKEN,
    part_query_ph=DEFAULT_PART_QUERY_TOKEN,
):
    """ candidate obj&part categories will be replaced by the actual categories later as they are dependent on the dataset
    """

    prefix_inst = f"This is an image {image_token_ph}, and please do object-part parsing on the objects on the image. \nGiven a list of object categories (e.g. {obj_cat_id}bicycle{obj_cat_id}, {obj_cat_id}bird{obj_cat_id}) and a list of part categories (e.g. {part_cat_id}bicycle's wheel{part_cat_id},  {part_cat_id}bicycle's saddle{part_cat_id},  {part_cat_id}bird's wing {part_cat_id}) and the object queries (e.g. {obj_query_id}some feature vector1{obj_query_id},  {obj_query_id}some feature vector2{obj_query_id}), please output the part queries following each object query (e.g. object {obj_query_id}some feature vector1{obj_query_id} with parts {part_query_id}some feature vector1{part_query_id},  {part_query_id}some feature vector2{part_query_id}). The output part queries will be used to predict the part masks and the part categories. Note {obj_cat_id} and {part_cat_id} are the delimiters for object categories and part categories, {obj_query_id} and {part_query_id} are the delimiters for object queries and part queries."
    
    prefix_inst_general = f"This is an image {image_token_ph}, and please do object-part parsing on the objects on the image. \nGiven a list of object categories (e.g. {general_cat_id}bicycle{general_cat_id}, {general_cat_id}bird{general_cat_id}) and a list of part categories (e.g. {general_cat_id}bicycle's wheel{general_cat_id},  {general_cat_id}bicycle's saddle{general_cat_id},  {general_cat_id}bird's wing {general_cat_id}) and the object queries (e.g. {obj_query_id}some feature vector1{obj_query_id},  {obj_query_id}some feature vector2{obj_query_id}), please output the part queries following each object query (e.g. object {obj_query_id}some feature vector1{obj_query_id} with parts {part_query_id}some feature vector1{part_query_id},  {part_query_id}some feature vector2{part_query_id}). The output part queries will be used to predict the part masks and the part categories. Note that {general_cat_id} is the delimiter for object categories and part categories, and {obj_query_id} and {part_query_id} are the delimiters for object queries and part queries."
    
    general_category_place_holders = f'\nThese all the candidate object and part categories: {general_category_ph}.'
    obj_category_place_holders = f'\nThese all the candidate object categories: {obj_categores_ph}.'
    part_category_place_holders = f'\nThese all the candidate part categories: {part_categores_ph}.'
    
    obj_query_prompts = "\nThese all the candidate object queries: " + "; ".join([f"object {obj_query_id}{obj_query_ph}{obj_query_id}"]*num_obs) + ".\n"
    
    part_querys_single_object = [ f"{part_query_id}{part_query_ph}{part_query_id}" ] * num_parts
    part_querys_single_object = ",".join(part_querys_single_object)
    obj_part_queries_single_obj = f" object {obj_query_id}{obj_query_ph}{obj_query_id} with parts: {part_querys_single_object}"
    obj_part_queries = [ obj_part_queries_single_obj ] * num_obs
    obj_part_queries = "; ".join(obj_part_queries)
    obj_query_answers = f"\nSure, the obj-part parsing result is here, all the candidate object queries followed by respetive part queryes: {obj_part_queries}.\n"
    
    # input_prompt = prefix_inst + obj_category_place_holders + part_category_place_holders + obj_query_prompts
    input_prompt = prefix_inst_general + general_category_place_holders + obj_query_prompts
    output_prompt = obj_query_answers
    
    print("input_prompt: ", input_prompt)
    print("output_prompt: ", output_prompt)
    
    sources = [[{'from': 'human', 'value':  input_prompt},
        {'from': 'gpt', 'value': output_prompt}]]
    
    return sources
    
def generate_prompts(
    num_obs, 
    num_parts, 
    general_cat_id = DEFAULT_GENERAL_CLS_IDENTIFIER,
    general_cat_id_end = DEFAULT_GENERAL_CLS_IDENTIFIER_END, 
    obj_cat_id = DEFAULT_OBJ_CLS_IDENTIFIER,
    part_cat_id = DEFAULT_PART_CLS_IDENTIFIER,
    obj_query_id = DEFAULT_OBJ_QUERY_IDENTIFIER,
    obj_query_id_end = DEFAULT_OBJ_QUERY_IDENTIFIER_END, 
    part_query_id = DEFAULT_PART_QUERY_IDENTIFIER,
    part_query_id_end = DEFAULT_PART_QUERY_IDENTIFIER_END, 
    image_token_ph=DEFAULT_IMAGE_TOKEN,
    general_category_ph=DEFAULT_GENERAL_CLS_TOKEN,
    obj_categores_ph=DEFAULT_OBJ_CLS_TOKEN,
    part_categores_ph=DEFAULT_PART_CLS_TOKEN,
    obj_query_ph=DEFAULT_OBJ_QUERY_TOKEN,
    part_query_ph=DEFAULT_PART_QUERY_TOKEN,
    mode = 'learnable_query',
):
    if mode == 'learnable_query':
        """ 
        simplified version compared to generate_prompts_archive to save space
        candidate obj&part categories will be replaced by the actual categories later as they are dependent on the dataset
        """
        prefix_inst_general = f"Please do object-part parsing on the {image_token_ph}. Given a list of object and part categories starting with {general_cat_id} and ending with {general_cat_id_end}(e.g. {general_cat_id}bicycle/bird/.../bicycle's wheel/bicycle's saddle/bird's wing{general_cat_id_end}), please output the part queries following each object query in the format: '{obj_query_id}obj_query{obj_query_id_end} with {part_query_id}part_querypart_query...part_query{part_query_id_end}."
        # " Each object query starts with {obj_query_id} and ends with {obj_query_id_end} and parts' queries start with {part_query_id} and end with {part_query_id_end}. The output part queries are used to predict the parts' masks and categories."
        
        general_category_place_holders = f'\nThese all the candidate object and part categories: {general_category_ph}.'
        
        obj_query_prompts = "\nThese all the candidate object queries: " + "; ".join([f"object {obj_query_id}{obj_query_ph}{obj_query_id}"]*num_obs) + ".\n"
        
        part_querys_single_object = part_query_id + "".join([part_query_ph]*num_parts) + part_query_id_end
        obj_part_queries_single_obj = f" object {obj_query_id}{obj_query_ph}{obj_query_id_end} with {part_querys_single_object}"
        obj_part_queries = [ obj_part_queries_single_obj ] * num_obs
        obj_part_queries = "; ".join(obj_part_queries)
        obj_query_answers = f"\nSure, result are: {obj_part_queries}.\n"
        
        # input_prompt = prefix_inst + obj_category_place_holders + part_category_place_holders + obj_query_prompts
        input_prompt = prefix_inst_general + general_category_place_holders
        output_prompt = obj_query_answers
        
        # print("input_prompt: ", input_prompt)
        # print("output_prompt: ", output_prompt)
        
        sources = [[{'from': 'human', 'value':  input_prompt},
            {'from': 'gpt', 'value': output_prompt}]]
    elif mode == 'clip_query':
        """
        use clip-based queries following the object queries to form obj-part queries
        """
        prefix_inst_general = f"Please do object-part parsing on the {image_token_ph}. For each object, you will be given a list of object-part queries in format of {part_query_id}part_querypart_query...part_query{part_query_id_end}, please refine the quries so that it can be used for later part category and mask prediction."
        part_querys_single_object = part_query_id + part_query_ph + part_query_id_end
        obj_query_prompts = "\nThese all the candidate object-part queries: " + "; ".join([f"object with parts {part_querys_single_object}"]*num_obs) + ".\n"
        input_prompt = prefix_inst_general + obj_query_prompts
        output_prompt = ''
        sources = [[{'from': 'human', 'value':  input_prompt},
            {'from': 'gpt', 'value': output_prompt}]]
        print("input_prompt: ", input_prompt)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return sources

def generate_prompts_vlm(
    num_obs, 
    num_parts, 
    general_cat_id = DEFAULT_GENERAL_CLS_IDENTIFIER,
    general_cat_id_end = DEFAULT_GENERAL_CLS_IDENTIFIER_END, 
    obj_cat_id = DEFAULT_OBJ_CLS_IDENTIFIER,
    part_cat_id = DEFAULT_PART_CLS_IDENTIFIER,
    obj_query_id = DEFAULT_OBJ_QUERY_IDENTIFIER,
    obj_query_id_end = DEFAULT_OBJ_QUERY_IDENTIFIER_END, 
    part_query_id = DEFAULT_PART_QUERY_IDENTIFIER,
    part_query_id_end = DEFAULT_PART_QUERY_IDENTIFIER_END, 
    image_token_ph=DEFAULT_IMAGE_TOKEN,
    general_category_ph=DEFAULT_GENERAL_CLS_TOKEN,
    obj_categores_ph=DEFAULT_OBJ_CLS_TOKEN,
    part_categores_ph=DEFAULT_PART_CLS_TOKEN,
    obj_query_ph=DEFAULT_OBJ_QUERY_TOKEN,
    part_query_ph=DEFAULT_PART_QUERY_TOKEN,
    mode = 'learnable_query',
):
    if mode == 'learnable_query':
        # """ 
        # simplified version compared to generate_prompts_archive to save space
        # candidate obj&part categories will be replaced by the actual categories later as they are dependent on the dataset
        # """
        # prefix_inst_general = f"Please do object-part parsing on the provided image {image_token_ph}. Given a list of object and part categories starting with {general_cat_id} and ending with {general_cat_id_end}(e.g. {general_cat_id}bicycle/bird/.../bicycle's wheel/bicycle's saddle/bird's wing{general_cat_id_end}), please output the part queries following each object query in the format: '{obj_query_id}obj_query{obj_query_id_end} with {part_query_id}part_querypart_query...part_query{part_query_id_end}."
        # # " Each object query starts with {obj_query_id} and ends with {obj_query_id_end} and parts' queries start with {part_query_id} and end with {part_query_id_end}. The output part queries are used to predict the parts' masks and categories."
        
        # general_category_place_holders = f'\nThese all the candidate object and part categories: {general_category_ph}.'
        
        # obj_query_prompts = "\nThese all the candidate object queries: " + "; ".join([f"object {obj_query_id}{obj_query_ph}{obj_query_id}"]*num_obs) + ".\n"
        
        # part_querys_single_object = part_query_id + "".join([part_query_ph]*num_parts) + part_query_id_end
        # obj_part_queries_single_obj = f" object {obj_query_id}{obj_query_ph}{obj_query_id_end} with {part_querys_single_object}"
        # obj_part_queries = [ obj_part_queries_single_obj ] * num_obs
        # obj_part_queries = "; ".join(obj_part_queries)
        # obj_query_answers = f"\nThe result are: {obj_part_queries}.\n"
        
        # # input_prompt = prefix_inst + obj_category_place_holders + part_category_place_holders + obj_query_prompts
        # output_prompt = prefix_inst_general + general_category_place_holders + obj_query_answers
        # print("output_prompt: ", output_prompt)
        """ 
        simplified version compared to generate_prompts_archive to save space
        candidate obj&part categories will be replaced by the actual categories later as they are dependent on the dataset
        """
        prefix_inst_general = f"Please do object-part parsing on the provided image {image_token_ph}. Given a list of object queries from the object segmentation module and the initial part queries of each object as input, please output the part queries following each object query in the format: '{obj_query_id}obj_query{obj_query_id_end} with {part_query_id}part_querypart_query...part_query{part_query_id_end}."
        # " Each object query starts with {obj_query_id} and ends with {obj_query_id_end} and parts' queries start with {part_query_id} and end with {part_query_id_end}. The output part queries are used to predict the parts' masks and categories."
        
        # general_category_place_holders = f'\nThese all the candidate object and part categories: {general_category_ph}.'
        
        # obj_query_prompts = "\nThese all the candidate object queries: " + "; ".join([f"object {obj_query_id}{obj_query_ph}{obj_query_id}"]*num_obs) + ".\n"
        
        part_querys_single_object = part_query_id + "".join([part_query_ph]*num_parts) + part_query_id_end
        obj_part_queries_single_obj = f" object {obj_query_id}{obj_query_ph}{obj_query_id_end} with {part_querys_single_object}"
        obj_part_queries = [ obj_part_queries_single_obj ] * num_obs
        obj_part_queries = "; ".join(obj_part_queries)
        obj_query_answers = f"\nThe result are: {obj_part_queries}.\n"
        
        # input_prompt = prefix_inst + obj_category_place_holders + part_category_place_holders + obj_query_prompts
        output_prompt = prefix_inst_general + obj_query_answers
        print("output_prompt: ", output_prompt)
    elif mode == 'clip_query':
        """
        use clip-based queries following the object queries to form obj-part queries
        """
        prefix_inst_general = f"Please do object-part parsing on the provided image {image_token_ph}. For each object, you will be given a list of object-part queries in format of {part_query_id}part_querypart_query...part_query{part_query_id_end}, please refine the quries so that it can be used for later part category and mask prediction."
        part_querys_single_object = part_query_id + part_query_ph + part_query_id_end
        obj_query_prompts = "\nThese all the candidate object-part queries: " + "; ".join([f"object with parts {part_querys_single_object}"]*num_obs) + ".\n"
        output_prompt = prefix_inst_general + obj_query_prompts
        print("output_prompt: ", output_prompt)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return output_prompt

def tokenizer_conversation_vlm(prompt, processor, 
                           image_token=DEFAULT_IMAGE_TOKEN,
                           image_index=IMAGE_TOKEN_INDEX,
                           general_cls_token=DEFAULT_GENERAL_CLS_TOKEN,
                           general_cls_index=GENERAL_CLASS_INDEX,
                           obj_cls_token=DEFAULT_OBJ_CLS_TOKEN,
                           obj_cls_index=OBJ_CLASS_INDEX,
                           part_cls_token=DEFAULT_PART_CLS_TOKEN,
                           part_cls_index=OBJ_PART_CLASS_INDEX,
                           obj_query_token=DEFAULT_OBJ_QUERY_TOKEN,
                           obj_query_index=OBJ_QUERY_INDEX,
                           part_query_token=DEFAULT_PART_QUERY_TOKEN,
                           part_query_index=OBJ_PART_QUERY_INDEX, 
                           return_tensors=None):
    special_token_map = {
        image_token: image_index,
        general_cls_token: general_cls_index,
        obj_cls_token: obj_cls_index,
        part_cls_token: part_cls_index,
        obj_query_token: obj_query_index,
        part_query_token: part_query_index,
    }
        
    split_texts = '(' + '|'.join(special_token_map.keys()) + ')'
    print("split_texts: ", split_texts)
    input_ids_batch = []

    input_ids = []
    prompt_chunks = re.split(split_texts, prompt)
    for chunk in prompt_chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        else:
            input_ids.extend(processor.tokenizer.encode(chunk, add_special_tokens=False))
    input_ids = torch.tensor(input_ids, dtype=torch.long).squeeze()
    
    return input_ids

def build_category_hiearachy(category_names, part_cat_indices, obj_parts_num = None):
    obj_part_hierarchy = {'id': {}, 'class_name': {}, 'obj_parts_num': {}}
    # print("category_names: ", category_names)
    obj_cat_indices = [i for i, _ in enumerate(category_names) if i not in part_cat_indices]
    obj_cats = [category_names[i] for i in obj_cat_indices]
    # obj_cats = category_names[obj_cat_indices]
    part_cats = [category_names[i] for i in part_cat_indices]
    
    num_parts_assigned = 0
    for obj_id in obj_cat_indices:
        obj_cat = category_names[obj_id]
        for par_id in part_cat_indices:
            part_cat = category_names[par_id]
            
            if obj_cat in part_cat:
                if obj_parts_num is None:
                    # print(" assigning {} to {}".format(part_cat, obj_cat))
                    if obj_cat not in obj_part_hierarchy['class_name']:
                        obj_part_hierarchy['class_name'][obj_cat] = []
                        obj_part_hierarchy['id'][obj_id] = []
                        obj_part_hierarchy['obj_parts_num'][obj_id] = {}
                    obj_part_hierarchy['class_name'][obj_cat].append(part_cat)
                    obj_part_hierarchy['id'][obj_id].append(par_id)
                    obj_part_hierarchy['obj_parts_num'][obj_id][par_id] = 3
                else:
                    if obj_cat not in obj_part_hierarchy['class_name']:
                        obj_part_hierarchy['class_name'][obj_cat] = []
                        obj_part_hierarchy['id'][obj_id] = []
                        obj_part_hierarchy['obj_parts_num'][obj_id] = {}
                    obj_part_hierarchy['class_name'][obj_cat].append(part_cat)
                    obj_part_hierarchy['id'][obj_id].append(par_id)
                    obj_part_hierarchy['obj_parts_num'][obj_id][par_id] = obj_parts_num[obj_id][par_id]
                
    return obj_part_hierarchy
    
# debug in main, if __name__ == "__main__", print output of formulate_prompts and preprocess_llama2:

if __name__ == "__main__":
    from transformers import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    num_obs = 50
    num_parts = 10

    sources = generate_prompts_vlm(num_obs, num_parts, mode = 'clip_query')
    # print("sources")
    
    
