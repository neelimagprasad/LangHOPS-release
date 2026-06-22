# Copyright (c) 2024 ByteDance. All Rights Reserved.
# PartGLEE Model.
# PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects (ECCV 2024)
# https://arxiv.org/abs/2407.16696

import torch
import torch.nn.functional as F
from torch import nn, os

from detectron2.modeling import build_backbone
from ..language import build_language_encoder

from .pixel_decoder.maskdino_encoder import build_pixel_decoder
from .transformer_decoder.maskdino_decoder import build_transformer_decoder
from .transformer_decoder.maskdino_part_decoder import build_part_transformer_decoder
from timm.models.layers import trunc_normal_
from transformers import CLIPTokenizer,CLIPTextModel
from .vos_utils import masks_to_boxes, FeatureFuser
from .transformer_decoder.transformer import QFormer, QFormerDecoderLayer
from ._CONSTANTS import OBJECT_LEVEL_DATASETS
from ._INFERENCE_CONSTANTS import TEST_TOPK_PER_IMAGE, DATASET_SPECIFIC_OBJECT_NUMS, DATASET_SPECIFIC_PART_NUMS, DATASET_SPECIFIC_CATEGORIES, PART_DATASETS_OBJECT_CATEGORY_INDEX, PART_DATASETS_PART_CATEGORY_INDEX, TEST_PART_ONLY_DATASETS, DATASET_PART_NUMS

from .language_models import conversation as conversation_lib
from .language_models.image_projector import ResNetSwin
from .language_models.llm_preprocessing import IGNORE_INDEX, IMAGE_TOKEN_INDEX, GENERAL_CLASS_INDEX, OBJ_CLASS_INDEX, OBJ_PART_CLASS_INDEX, OBJ_QUERY_INDEX, OBJ_PART_QUERY_INDEX, ADDITIONAL_TOKENS, preencode_dataset_names, generate_prompts, tokenizer_conversation, generate_prompts_vlm, tokenizer_conversation_vlm, build_category_hiearachy
from peft import LoraConfig, get_peft_model

def rand_sample(x, max_len):
    if x.shape[1] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[1])[:max_len]
        return x[:,rand_idx]


def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate

class PartGLEE_Model(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    def __init__(self, cfg, matcher, device, video_info, contras_mean, unify_object_part, detach_object_queries = False, data_classes_dict = None):
        super().__init__()
        self.cfg = cfg
        self.freeze_backbone = cfg.FREEZE_BACKBONE
        self.freeze_pixel_decoder = cfg.FREEZE_PIXEL_DECODERE
        self.matcher = matcher
        self.backbone = build_backbone(cfg)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.detach_object_queries = detach_object_queries
        output_channels = [v for k,v in self.backbone._out_feature_channels.items()]
        if cfg.MODEL.VISUAL_PROMPT:
            self.sot_fuser = FeatureFuser(output_channels[-3:], 256)
        
        self.text_encode_type = cfg.MODEL.TEXT.ARCH
        if cfg.MODEL.TEXT.ARCH == 'clip_frozen':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
        elif cfg.MODEL.TEXT.ARCH == 'clip_unfreeze':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
            self.text_encode_type = 'clip_frozen'
        elif cfg.MODEL.TEXT.ARCH == 'clip_teacher':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.text_encoder_teacher = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder_teacher.parameters():
                p.requires_grad = False
            self.text_encoder_teacher.eval()
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
        

        # self.lang_encoder = None     
        self.pixel_decoder = build_pixel_decoder(cfg, self.backbone.output_shape())
        if self.freeze_pixel_decoder:
            for param in self.pixel_decoder.parameters():
                param.requires_grad = False
        
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.use_early_fusion = cfg.MODEL.USE_EARLYFUSION
        self.object_predictor = build_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True,)
        self.part_predictor = build_part_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True, is_part_decoder=True)
        
        # Unify object-part tasks
        self.unify_object_part = unify_object_part
        self.use_qformer = cfg.MODEL.MaskDINO.Q_FORMER
        
        if self.unify_object_part:
            # Hyperparameters for Q-Former
            num_object_queries = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
            self.num_object_queries = num_object_queries
            num_part_queries = cfg.MODEL.MaskDINO.NUM_PART_QUERIES
            self.num_part_queries = num_part_queries
            hidden_dim = cfg.MODEL.MaskDINO.HIDDEN_DIM
            self.num_decoder_layer = cfg.MODEL.MaskDINO.OBJECT_PART_DECODER_LAYERS
            self.topk_object_queries_num = cfg.MODEL.MaskDINO.TOPK_OBJECT_QUERIES
            self.part_queries_feat = nn.Embedding(num_part_queries, hidden_dim)
            
            nhead = cfg.MODEL.MaskDINO.NHEADS
            dim_feedforward = cfg.MODEL.MaskDINO.DIM_FEEDFORWARD
            dropout = 0.0
            activation = 'relu'
            self.part_queries_pos = nn.Embedding(num_part_queries, hidden_dim)
            self.obj_queries_pos = nn.Embedding(num_object_queries, hidden_dim)
            qformer_layer = QFormerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
            qformer_norm = nn.LayerNorm(hidden_dim)
            self.qformer = QFormer(qformer_layer, self.num_decoder_layer, qformer_norm, return_intermediate=False)
            
        self.device = device
        self.to(device)
        
        self.visualize = cfg.MODEL.MaskDINO.TEST.VISUALIZE
        
        self.test_topk_per_image = TEST_TOPK_PER_IMAGE
        self.test_object_inst_num = DATASET_SPECIFIC_OBJECT_NUMS
        self.test_part_inst_num = DATASET_SPECIFIC_PART_NUMS
        self.num_classes = DATASET_SPECIFIC_CATEGORIES
        self.object_category_index_mapper = PART_DATASETS_OBJECT_CATEGORY_INDEX
        self.part_category_index_mapper = PART_DATASETS_PART_CATEGORY_INDEX
        
        part_level_tasks = [key[:-5] if key.endswith('_part') else key for key in PART_DATASETS_PART_CATEGORY_INDEX.keys()]
        self.dataset_part_nums = DATASET_PART_NUMS
        print("part_level_tasks: ", part_level_tasks)
        
        self.use_clip_part_nums_prior = cfg.MODEL.LLM.PART_CLIP_QUERY_PRIOR_PART_NUMS
        self.cat_hierachies = {}
        for key in part_level_tasks:
            if key in data_classes_dict:
                if self.use_clip_part_nums_prior and key in self.dataset_part_nums:
                    obj_parts_num = self.dataset_part_nums[key]
                    print("{} part nums: {}".format(key, obj_parts_num))
                else:
                    obj_parts_num = None
                self.cat_hierachies[key] = build_category_hiearachy(data_classes_dict[key], self.part_category_index_mapper[key + '_part'], obj_parts_num)
        
        self.object_level_datasets = OBJECT_LEVEL_DATASETS
        self.test_part_only_datasets = TEST_PART_ONLY_DATASETS
        
        self.video_info = video_info
        self.contras_mean = contras_mean
        self.track_loss_version = cfg.MODEL.TRACK_VERSION

        # for visual prompt
        hidden_dim = 256
        self.max_spatial_len = [512,512,512,512]
        self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(4)])
        trunc_normal_(self.mask_sptial_embed[0], std=.02)
        trunc_normal_(self.mask_sptial_embed[1], std=.02)
        trunc_normal_(self.mask_sptial_embed[2], std=.02)
        trunc_normal_(self.mask_sptial_embed[3], std=.02)
        # learnable positive negative indicator
        self.pn_indicator = nn.Embedding(2, hidden_dim)
        
        # llm
        self.data_classes_dict = data_classes_dict
        self.init_llm(cfg)
    
    def init_llm(self, cfg):
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([str(x) in str(name) for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
                # print("name: ", name)
            # print("lora_target_modules: ", lora_target_modules)
            # if any([x in name for x in lora_target_modules]):
            #     print(" x: {}, name: {}".format(x, name))
            return sorted(list(lora_module_names))
        self.use_llm = cfg.MODEL.LLM.USE_LLM
        self.llm_hidden_dim = cfg.MODEL.LLM.HIDDEN_SIZE
        self.llm_debug = cfg.MODEL.LLM.DEBUG
        self.llm_id = cfg.MODEL.LLM.LLM_TYPE
        self.llm_path = cfg.MODEL.LLM.LLM_PATH
        self.part_query_mode = cfg.MODEL.LLM.PART_QUERY_MODE
        self.clip_emb_part_atten_mask = cfg.MODEL.LLM.CLIP_EMB_PART_ATTEN_MASK
        self.part_query_emb_mode = cfg.MODEL.LLM.PART_CLS_EMBED 
        self.mm_input_dim = cfg.MODEL.LLM.IMG_PROJ_INDIMS
        conversation_type = cfg.MODEL.LLM.LLM_CONVERSATION_TYPE
        assert conversation_type in conversation_lib.conv_templates, f"Conversation type {conversation_type} not found in conversation templates."
        self.conversation_template = conversation_lib.conv_templates[conversation_type]
        
        # use_qformer and use_llm cannot be true at the same time
        assert not (self.use_qformer and self.use_llm), "use_qformer and use_llm cannot be true at the same time"
        
        if self.use_llm:
            self.use_vlm = cfg.MODEL.LLM.VLM.IS_VLM
            
            if not self.use_vlm:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                # if the path is provided, use path, otherwise use llm_id
                if os.path.exists(self.llm_path):
                    print("Loading LLM from path:", self.llm_path)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_path)
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
                else:
                    print("Loading LLM from id:", self.llm_id)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_id)
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_id)
                for token in ADDITIONAL_TOKENS:
                    self.llm_tokenizer.add_tokens(token)
                print(" max length of input:", self.llm_model.model.config.max_position_embeddings)
                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
                # assume token embedding func is frozen
                self.llm_token_embed_func = self.llm_model.model.embed_tokens
                # print("dir(self.llm_model): ", dir(self.llm_model))
                # print("dir(self.llm_model.model): ", dir(self.llm_model.model))
            else:
                from transformers import  PaliGemmaProcessor, PaliGemmaForConditionalGeneration, AutoTokenizer
                if self.llm_id == "google/paligemma2-3b-pt-448":
                    self.llm_model = PaliGemmaForConditionalGeneration.from_pretrained(self.llm_id)
                    print("model device: ", next(self.llm_model.parameters()).device)
                    print("self.device: ", self.device)
                    self.llm_processor = PaliGemmaProcessor.from_pretrained(self.llm_id)
                    for token in ADDITIONAL_TOKENS:
                        self.llm_processor.tokenizer.add_tokens(token)
                    self.llm_model.resize_token_embeddings(len(self.llm_processor.tokenizer))
                    self.llm_tokenizer = self.llm_processor.tokenizer
                    self.llm_token_embed_func = self.llm_model.get_input_embeddings()
                else:
                    raise ValueError("llm_id {} not supported! ".format(self.llm_id)) 
                
                
            # print("self.llm_model.model.embed_tokens: ", self.llm_model.model.embed_tokens.shape)
            # model with lora
            lora_r = cfg.MODEL.LLM.LORA.R
            lora_alpha = cfg.MODEL.LLM.LORA.ALPHA
            lora_dropout = cfg.MODEL.LLM.LORA.DROPOUT
            target_modules = cfg.MODEL.LLM.LORA.TARGET_MODULES
            lora_target_modules = find_linear_layers(
                self.llm_model, target_modules
            )
            # print module names
            name_list = []
            for name, module in self.llm_model.named_modules():
                name_list.append(name)
            # print(" modules of llm: ", name_list)
            print("lora_target_modules:", lora_target_modules)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            # print("self.llm_model.model.embed_tokens: ", self.llm_model.base_model.model.model.embed_tokens)
            self.llm_model.print_trainable_parameters()
                
            # visual projector
            self.vision_projector = ResNetSwin(input_dim=self.mm_input_dim, out_dim=self.llm_hidden_dim)
            
            self.topk_object_queries_num = cfg.MODEL.MaskDINO.TOPK_OBJECT_QUERIES
            self.num_part_queries = cfg.MODEL.MaskDINO.NUM_PART_QUERIES
            # create prompt template
            
            if self.use_vlm:
                prompt = generate_prompts_vlm(self.topk_object_queries_num, self.num_part_queries, mode=self.part_query_mode)
                self.input_ids_template = [tokenizer_conversation_vlm(prompt, self.llm_processor)]
            else:
                self.prompt_template = generate_prompts(self.topk_object_queries_num, self.num_part_queries, mode=self.part_query_mode)
                self.input_ids_template = tokenizer_conversation(self.prompt_template, self.conversation_template, self.llm_tokenizer, return_tensors='pt')
            # print("self.input_ids_template: ", self.input_ids_template)
            # save self.input_ids_template to text:
            # torch.set_printoptions(threshold=float('inf'))
            # with open("/work/yang_miao/PartGleeMount/train_pascalpart_to_partimagenet_llm_debug/input_ids_template.txt", "w") as f:
            #     f.write(str(self.input_ids_template))
            # pre-encode dataset names
            assert self.data_classes_dict is not None, "data_classes_dict must be provided when using llm"
            self.dataset_class_ids, self.dataset_class_ids_indices = preencode_dataset_names(self.data_classes_dict, self.llm_tokenizer)
            
            # obj and part queries 
            self.mask_hidden_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
            if self.part_query_mode == "learnable_query":
                self.part_queries_llm = nn.Embedding(self.num_part_queries, self.llm_hidden_dim)
                mask_hidden_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
                ## create a mlp to project the object-part query concatenation to the llm hidden dim, module list, linear layer, bn, relu, for 3 layers
                self.obj_part_queries_mapper = nn.Sequential(
                    nn.Linear(self.llm_hidden_dim*2, self.llm_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(self.llm_hidden_dim),
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                        nn.ReLU(), nn.BatchNorm1d(self.llm_hidden_dim),
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim)
                    )
                ## object query projector
                self.obj_queries_projector = nn.Linear(mask_hidden_dim, self.llm_hidden_dim) # [256 -> 2048]
                ## object-part query back projector to mask hidden dim
                self.obj_queries_back_mapper = nn.Sequential(
                    nn.Linear(self.llm_hidden_dim, mask_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(mask_hidden_dim),
                        nn.Linear(mask_hidden_dim, mask_hidden_dim)
                    )
                self.part_queries_back_mapper = nn.Sequential(
                    nn.Linear(self.llm_hidden_dim, mask_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(mask_hidden_dim),
                        nn.Linear(mask_hidden_dim, mask_hidden_dim)
                    )
                # class name embed pooling
                self.class_name_pooling = nn.AdaptiveAvgPool1d(output_size=1)
                self.class_name_embed_projector = nn.Sequential(
                    nn.Linear(self.llm_hidden_dim, mask_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(mask_hidden_dim),
                        nn.Linear(mask_hidden_dim, mask_hidden_dim),
                        nn.ReLU(), nn.BatchNorm1d(mask_hidden_dim),
                        nn.Linear(mask_hidden_dim, mask_hidden_dim)
                    )
            elif self.part_query_mode == "clip_query":
                clip_class_emb_dim =  cfg.MODEL.DIM_PROJ
                mask_hidden_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
                obj_part_concat_dim = mask_hidden_dim + clip_class_emb_dim
                self.obj_part_concat_dim = obj_part_concat_dim
                self.obj_part_queries_mapper = nn.Sequential(
                    nn.Linear(obj_part_concat_dim, self.llm_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(self.llm_hidden_dim),
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                        nn.ReLU(), nn.BatchNorm1d(self.llm_hidden_dim),
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim)
                    )
                self.part_queries_back_mapper = nn.Sequential(
                    nn.Linear(self.llm_hidden_dim, mask_hidden_dim), 
                        nn.ReLU(), nn.BatchNorm1d(mask_hidden_dim),
                        nn.Linear(mask_hidden_dim, mask_hidden_dim)
                    )

    
    def forward(self, images, prompts, task, targets=None, batch_name_list=None, is_train=True, criterion=None, custom_object_categories_idx=None, custom_part_categories_idx=None, images_vlm = None):
        extra =  {}
        early_semantic = None
        if self.text_encode_type == 'clip_frozen':
            if task not in ['grounding','rvos']:
                assert batch_name_list
                classes_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(classes_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")
                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']
                token_x = torch.matmul(token_x, self.lang_projection.contiguous())
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs, 768)
                extra['class_embeddings'] = lang_feat_pool
                dist_loss =  (lang_feat_pool*0).sum()
                if self.use_early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask}
        
        elif self.text_encode_type == "clip_teacher":
            if task not in ['grounding','rvos']:
                assert batch_name_list
                classes_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(classes_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")

                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']

                valid_mask = tokenized['attention_mask'].bool()
                with torch.no_grad():
                    token_x_teacher = self.text_encoder_teacher(*texts)['last_hidden_state']
                dist_loss =  F.mse_loss(token_x[valid_mask], token_x_teacher[valid_mask] )
                # token_x = token_x @ self.lang_projection
                token_x = torch.matmul(token_x, self.lang_projection.contiguous())
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs,  768)
                extra['class_embeddings'] = lang_feat_pool 
                if self.use_early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 

        if isinstance(images,torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        
        mask_features, _, multi_scale_features, zero_loss = self.pixel_decoder.forward_features(features, masks=None, early_fusion = early_semantic)
        
        # mask_features is the dense features Conv2d(multi_scale_features[-1]), multi_scale_features is multi-scale features including mask_features
        # print("multi_scale_features.len:", len(multi_scale_features))
        # for level in range(len(multi_scale_features)):
        #     print("multi_scale_features[{}].shape:".format(level), multi_scale_features[level].shape)
        # print("mask_features.len:", len(mask_features))
        # print("mask_features.shape:", mask_features.shape)
        # multi_scale_features.len: 4
        # multi_scale_features[0].shape: torch.Size([1, 256, 128, 128])
        # multi_scale_features[1].shape: torch.Size([1, 256, 64, 64])
        # multi_scale_features[2].shape: torch.Size([1, 256, 32, 32])
        # multi_scale_features[3].shape: torch.Size([1, 256, 16, 16])
        # mask_features.shape: torch.Size([1, 256, 200, 328])
        
        ## ensure all params in loss caculation 
        if early_semantic:
            params_zero_loss = zero_loss + (self.pn_indicator.weight*0).sum()
        else:
            zero_loss = 0
            params_zero_loss = zero_loss + (self.pn_indicator.weight*0).sum()
            
        for p in self.mask_sptial_embed:
            params_zero_loss += (p*0).sum()

        params_zero_loss += (self.object_predictor.coco_label_enc.weight*0).sum()  +\
        (self.object_predictor.obj365_label_enc.weight*0).sum() +\
        (self.object_predictor.vg_label_enc.weight*0).sum() +\
        (self.object_predictor.grounding_label_enc.weight*0).sum() +\
        (self.object_predictor.ytvis19_label_enc.weight*0).sum() +\
        (self.object_predictor.ytvis21_label_enc.weight*0).sum() +\
        (self.object_predictor.ovis_label_enc.weight*0).sum() +\
        (self.object_predictor.uvo_label_enc.weight*0).sum() +\
        (self.object_predictor.bdd_det.weight*0).sum() +\
        (self.object_predictor.bdd_inst.weight*0).sum()
        # print("is_train: {}, unify_object_part: {}".format(is_train, self.unify_object_part))
        if is_train and self.unify_object_part:
            part_losses = {}
            topk = self.topk_object_queries_num
            if task not in self.object_level_datasets or task == 'part_classification':
                object_targets, part_targets = targets
                if task == 'part_classification':
                    object_level_task = 'part_classification_object'
                    part_level_task = 'part_classification_part'
                else:
                    object_level_task = task + '_object'
                    part_level_task = task + '_part'
            else:
                object_targets = targets
                part_targets = [{"labels": torch.tensor([]).to(object_targets[i]['labels'].device), "boxes": torch.empty(size=(0,4), device=object_targets[i]['labels'].device), "masks": None} for i in range(len(object_targets))]
                object_level_task = task
                part_level_task = task
                
            assert criterion is not None, 'The training phase must acquire criterion to perform matching and obtain the matched indices'

            object_predictor_kwargs = {}
            if object_level_task == 'part_classification_object' and custom_object_categories_idx is not None:
                object_predictor_kwargs['custom_object_categories_idx'] = custom_object_categories_idx

            object_outputs, object_mask_dict, object_queries, src_flatten = self.object_predictor(
                multi_scale_features, mask_features, extra=extra, task=object_level_task, masks=None, targets=object_targets, **object_predictor_kwargs
            )
            
            fake_object_track_loss = (object_outputs['pred_track_embed']*0).sum()
            # Perform matching in the object-level instances
            object_losses, matched_indices = criterion(object_outputs, object_targets, object_mask_dict, object_level_task)
            object_losses.update({"track_loss":fake_object_track_loss})
            object_losses.update({"dist_loss":dist_loss+params_zero_loss})
            
            # Q-Former
            # In order to generate part queries for each object queries, we transform the dim of object queries into [bs*nq,1,c=256]
            num_part_queries = self.num_part_queries
            if self.use_qformer:
                bs, num_object_queries, hidden_dim = object_queries.shape
                if self.detach_object_queries:
                    object_queries = object_queries.detach()     
                topk_object_queries_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk, 1)[1]
                # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
                topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                topk_object_queries = topk_object_queries.flatten(0,1).unsqueeze(1)   #[bs*nq,1,c=256]
                bsnq = topk_object_queries.shape[0]
                
                part_queries = self.part_queries_feat.weight
                part_queries = part_queries.repeat(bsnq, 1, 1)
                
                part_queries_pos = self.part_queries_pos.weight.repeat(bsnq, 1, 1)
                object_queries_pos = torch.gather(self.obj_queries_pos.weight.repeat(bs, 1, 1), 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                object_queries_pos = object_queries_pos.flatten(0,1).unsqueeze(1)     #[bs*nq,1,c=256]
                part_queries = self.qformer(
                    tgt=part_queries.transpose(0,1).contiguous(),
                    memory=topk_object_queries.transpose(0,1).contiguous(),
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=object_queries_pos.transpose(0,1).contiguous(),
                    query_pos=part_queries_pos.transpose(0,1).contiguous(),
                )       # part_queries:[num_part_queries, bs*num_object_queries, c=256]
                
                part_queries = part_queries.reshape(bs, topk*self.num_part_queries, hidden_dim).contiguous()
                
                part_predictor_kwargs = {
                    'part_queries': part_queries,
                    'topk': topk,
                    'num_part_queries': topk*self.num_part_queries,
                }
                if part_level_task == 'part_classification_part' and custom_part_categories_idx is not None:
                    part_predictor_kwargs['custom_part_categories_idx'] = custom_part_categories_idx

                # Part-level Decoder
                part_outputs, part_mask_dict = self.part_predictor(
                    multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, targets=part_targets, **part_predictor_kwargs
                )
                
                # Perform matching in the part-level instances
                fake_part_track_loss = (part_outputs['pred_track_embed']*0).sum()
                if task in self.object_level_datasets and task != 'part_classification':
                    fake_task = task + '_fake'
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, fake_task, object_outputs, topk_object_queries_indices)
                else:
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, part_level_task, object_outputs, topk_object_queries_indices)
                part_losses.update({"track_loss":fake_part_track_loss})
                part_losses.update({"dist_loss":dist_loss+params_zero_loss})
                
            elif self.use_llm:
                bs, num_object_queries, hidden_dim = object_queries.shape  
                logits_max, max_indices = object_outputs['pred_logits'].max(-1) # [bs, nq]
                topk_object_queries_indices = torch.topk(logits_max, topk, 1)[1] # [bs, topk]
                topk_classes = max_indices.gather(1, topk_object_queries_indices) # [bs, topk]
                
                if self.detach_object_queries:
                    object_queries = object_queries.detach()  
                
                # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
                topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                
                if self.use_vlm:
                    num_images = len(images_vlm)
                    # image_list = [images[i] for i in range(num_images)]
                    text_place_holder = [" "] * num_images
                    pixel_values = self.llm_processor(images = images_vlm, text = text_place_holder, return_tensors="pt").to(self.device)['pixel_values']  # text id will be precessed later 
                    pixel_values
                    vision_outputs = self.llm_model.base_model.model.vision_tower(pixel_values=pixel_values)
                    vision_outputs = vision_outputs.last_hidden_state
                    # print("vision_outputs: ", vision_outputs.shape)
                    vision_embed = self.llm_model.base_model.model.multi_modal_projector(vision_outputs)
                    # image_embeds = vision_outputs.last_hidden_state  # shape: [B, N, D]
                    # print("vision_embed: ", vision_embed.shape)
                else:
                    vision_embed = None
                
                llm_input = self.prepare_multimodal_llm_input(
                    task, topk_object_queries, multi_scale_features, topk_classes, extra,
                    images_embed = vision_embed, batch_names = batch_name_list
                )
                new_input_embeds_batch = llm_input['new_input_embeds_batch']
                attention_mask_batch = llm_input['attention_mask']
                # print("attention_mask_batch.sum(-1): ", attention_mask_batch.sum(-1))
                if not self.use_vlm:
                    outputs = self.llm_model(
                        input_ids=None,
                        attention_mask=attention_mask_batch,
                        past_key_values=None,
                        inputs_embeds=new_input_embeds_batch,
                        use_cache=None,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True
                    )
                else:
                    pixel_values = llm_input['pixel_values']
                    outputs = self.llm_model(
                        input_ids=None,
                        pixel_values = pixel_values, 
                        attention_mask=attention_mask_batch,
                        past_key_values=None,
                        inputs_embeds=new_input_embeds_batch,
                        use_cache=None,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True
                    )
                if self.part_query_mode == 'learnable_query':
                    hidden_states = outputs.hidden_states[-1]
                    cls_embed_indices_batch = llm_input['cls_embed_indices_batch']
                    part_query_indices_batch = llm_input['part_query_indices_batch']
                    # class_name_embedding = self.get_class_name_embedding( [hidden_states[0]], [cls_embed_indices_batch[0]]) # assume same dataset on one batch 
                    # _, num_classes, hidden_dim = class_name_embedding.shape # _ = 1, as we only use first batch item for class_name_embedding
                    # class_name_embedding = class_name_embedding.contiguous().view(num_classes, hidden_dim)
                    # class_name_embedding = self.class_name_embed_projector(class_name_embedding)
                    # class_name_embedding = class_name_embedding.contiguous().view(num_classes, -1).contiguous()
                    obj_part_queries_dict = self.get_seg_query(hidden_states, part_query_indices_batch, self.clip_emb_part_atten_mask)
                    obj_part_queries = obj_part_queries_dict['seg_query']
                    obj_part_query_atten_mask = obj_part_queries_dict['atten_mask']
                    num_part_queries = obj_part_queries.shape[1]
                    # squeeze the obj_part_queries to [bs * num_topk_queries*num_part_queries, C]
                    obj_part_queries = obj_part_queries.contiguous().view(bs*num_part_queries, -1)
                    part_queries = self.part_queries_back_mapper(obj_part_queries) # [bs, num_topk_objects, num_part_queries, 256]
                    part_queries = part_queries.view(bs, self.topk_object_queries_num, self.num_part_queries, -1)
                    part_queries = part_queries.reshape(bs, num_part_queries, -1).contiguous()
                    
                    
                    if self.part_query_emb_mode == 'learnable':
                        if part_level_task in self.part_category_index_mapper:
                            part_idxs = self.part_category_index_mapper[part_level_task]
                            part_idxs = torch.tensor(part_idxs, device = class_name_embedding.device)
                            class_embeddings = extra['class_embeddings'].clone()
                            class_embeddings[part_idxs, :] = class_name_embedding[part_idxs, :].float()
                            extra['class_embeddings'] = class_embeddings
                        else:
                            pass

                    elif self.part_query_emb_mode == 'clip':
                        pass
                    else:
                        raise ValueError("part_query_emb_mode {} not implemented! ".format(self.part_query_emb_mode)) 

                    # print("torch.isnan(hidden_states).any(): ", torch.isnan(hidden_states).any())
                    # print("torch.isnan(obj_part_queries).any(): ", torch.isnan(hidden_states).any())
                    # print("torch.isnan(part_queries).any(): ", torch.isnan(part_queries).any())
                    # print("torch.isnan(class_name_embedding).any(): ", torch.isnan(class_name_embedding).any())
                    
                elif self.part_query_mode == 'clip_query':
                    hidden_states = outputs.hidden_states[-1]
                    
                    part_query_indices_batch = llm_input['part_query_indices_batch']
                    obj_part_queries_dict = self.get_seg_query(hidden_states, part_query_indices_batch, self.clip_emb_part_atten_mask)
                    obj_part_queries = obj_part_queries_dict['seg_query']
                    obj_part_query_atten_mask = obj_part_queries_dict['atten_mask']
                    num_part_queries = obj_part_queries.shape[1]
                    # squeeze the obj_part_queries to [bs * num_topk_queries*num_part_queries, C]
                    obj_part_queries = obj_part_queries.contiguous().view(bs*num_part_queries, -1)
                    part_queries = self.part_queries_back_mapper(obj_part_queries) # [bs, num_topk_objects, num_part_queries, 256]
                    part_queries = part_queries.contiguous().view(bs, num_part_queries, -1)
                    
                    # print("part_queries: ", part_queries.shape)
    
                else:
                    raise ValueError("part_query_mode must be learnable_query or clip_query")  
                
                # Part-level Decoder
                part_outputs, part_mask_dict = self.part_predictor(multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, targets=part_targets, part_queries=part_queries, topk=topk, num_part_queries=num_part_queries, part_query_attn_mask = obj_part_query_atten_mask)
                
                # Perform matching in the part-level instances
                fake_part_track_loss = (part_outputs['pred_track_embed']*0).sum()
                if task in self.object_level_datasets:
                    fake_task = task + '_fake'
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, fake_task, object_outputs, topk_object_queries_indices)
                else:
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, part_level_task, object_outputs, topk_object_queries_indices)
                part_losses.update({"track_loss":fake_part_track_loss})
                part_losses.update({"dist_loss":dist_loss+params_zero_loss})

            losses = object_losses
            
            if self.use_llm or self.use_qformer: 
                for key in part_losses.keys():    
                    losses.update({"part_" + key: part_losses[key]})
            return losses
        else:
            # Inference
            outputs = self.hierarchical_inference(batch_name_list, task, targets, multi_scale_features, mask_features, extra, custom_object_categories_idx, custom_part_categories_idx, images_vlm)
            
            return outputs
        
    def hierarchical_inference(self, batch_name_list, task, targets, multi_scale_features, mask_features, extra, custom_object_categories_idx, custom_part_categories_idx, images_vlm):
        topk = self.topk_object_queries_num
        
        if task not in self.object_level_datasets:
            object_level_task = task + '_object'
            part_level_task = task + '_part'
        else:
            object_level_task = part_level_task = task

        object_outputs, _, object_queries, src_flatten = self.object_predictor(multi_scale_features, mask_features, extra=extra, task=object_level_task, masks=None, targets=targets, custom_object_categories_idx=custom_object_categories_idx)
        # print("object_queries.shape:", object_queries.shape)
        part_outputs = None
        
        if task not in self.object_level_datasets:
            # Q-Former
            if self.use_qformer:
                # In order to generate part queries for each object queries, we transform the dim of object queries into [bs*nq,1,c=256]
                bs, num_object_queries, hidden_dim = object_queries.shape
                topk_object_queries_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk, 1)[1]
                # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
                topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                topk_object_queries = topk_object_queries.flatten(0,1).unsqueeze(1)   #[bs*nq,1,c=256]
                bsnq = topk_object_queries.shape[0]
                part_queries = self.part_queries_feat.weight.repeat(bsnq, 1, 1)

                part_queries_pos = self.part_queries_pos.weight.repeat(bsnq, 1, 1)
                object_queries_pos = torch.gather(self.obj_queries_pos.weight.repeat(bs, 1, 1), 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                object_queries_pos = object_queries_pos.flatten(0,1).unsqueeze(1)     #[bs*nq,1,c=256]
                part_queries = self.qformer(
                    tgt=part_queries.transpose(0,1).contiguous(),
                    memory=topk_object_queries.transpose(0,1).contiguous(),
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=object_queries_pos.transpose(0,1).contiguous(),
                    query_pos=part_queries_pos.transpose(0,1).contiguous(),
                )       # part_queries:[num_part_queries, bs*num_object_queries, c=256]
                    
                part_queries = part_queries.reshape(bs, topk*self.num_part_queries, hidden_dim).contiguous()
                num_part_queries = part_queries.shape[1]
                
                # Part-level Decoder
                part_outputs, _ = self.part_predictor(multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, \
                                                    part_queries=part_queries, topk=topk, num_part_queries=num_part_queries, custom_part_categories_idx=custom_part_categories_idx)
                
            elif self.use_llm:
                bs, num_object_queries, hidden_dim = object_queries.shape  
                logits_max, max_indices = object_outputs['pred_logits'].max(-1) # [bs, nq]
                topk_object_queries_indices = torch.topk(logits_max, topk, 1)[1] # [bs, topk]
                topk_classes = max_indices.gather(1, topk_object_queries_indices) # [bs, topk]
                
                # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
                topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                
                if self.use_vlm:
                    num_images = len(images_vlm)
                    # image_list = [images[i] for i in range(num_images)]
                    text_place_holder = [" "] * num_images
                    pixel_values = self.llm_processor(images = images_vlm, text = text_place_holder, return_tensors="pt").to(self.device)['pixel_values']  # text id will be precessed later 
                    pixel_values
                    vision_outputs = self.llm_model.base_model.model.vision_tower(pixel_values=pixel_values)
                    vision_outputs = vision_outputs.last_hidden_state
                    # print("vision_outputs: ", vision_outputs.shape)
                    vision_embed = self.llm_model.base_model.model.multi_modal_projector(vision_outputs)
                else:
                    vision_embed = None
                
                llm_input = self.prepare_multimodal_llm_input(
                    task, topk_object_queries, multi_scale_features, topk_classes, extra,
                    images_embed = vision_embed, batch_names = batch_name_list
                )
                
                new_input_embeds_batch = llm_input['new_input_embeds_batch']
                attention_mask_batch = llm_input['attention_mask']
                
                outputs = self.llm_model(
                    input_ids=None,
                    attention_mask=attention_mask_batch,
                    past_key_values=None,
                    inputs_embeds=new_input_embeds_batch,
                    use_cache=None,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True
                    )
                if self.part_query_mode == 'learnable_query':
                    hidden_states = outputs.hidden_states[-1]
                    
                    cls_embed_indices_batch = llm_input['cls_embed_indices_batch']
                    part_query_indices_batch = llm_input['part_query_indices_batch']
                    
                    # class_name_embedding = self.get_class_name_embedding( [hidden_states[0]], [cls_embed_indices_batch[0]]) # assume same dataset on one batch 
                    # _, num_classes, hidden_dim = class_name_embedding.shape # _ = 1, as we only use first batch item for class_name_embedding
                    # class_name_embedding = class_name_embedding.contiguous().view(num_classes, hidden_dim)
                    # class_name_embedding = self.class_name_embed_projector(class_name_embedding)
                    # class_name_embedding = class_name_embedding.contiguous().view(num_classes, -1)
                    obj_part_queries_dict = self.get_seg_query(hidden_states, part_query_indices_batch, self.clip_emb_part_atten_mask)
                    obj_part_queries = obj_part_queries_dict['seg_query']
                    obj_part_query_atten_mask = obj_part_queries_dict['atten_mask']
                    num_part_queries = obj_part_queries.shape[1]
                    # squeeze the obj_part_queries to [bs * num_topk_queries*num_part_queries, C]
                    obj_part_queries = obj_part_queries.contiguous().view(bs*num_part_queries, -1)
                    part_queries = self.part_queries_back_mapper(obj_part_queries) # [bs, num_topk_objects, num_part_queries, 256]
                    part_queries = part_queries.reshape(bs, num_part_queries, -1).contiguous()
                    if self.part_query_emb_mode == 'learnable':
                        if part_level_task in self.part_category_index_mapper:
                            part_idxs = self.part_category_index_mapper[part_level_task]
                            part_idxs = torch.tensor(part_idxs, device = class_name_embedding.device)
                            class_embeddings = extra['class_embeddings'].clone()
                            class_embeddings[part_idxs, :] = class_name_embedding[part_idxs, :].float()
                            extra['class_embeddings'] = class_embeddings
                        else:
                            pass
                    elif self.part_query_emb_mode == 'clip':
                        pass
                    else:
                        raise ValueError("part_query_emb_mode {} not implemented! ".format(self.part_query_emb_mode)) 
                    # print("torch.isnan(hidden_states).any(): ", torch.isnan(hidden_states).any())
                    # print("torch.isnan(obj_part_queries).any(): ", torch.isnan(hidden_states).any())
                    # print("torch.isnan(part_queries).any(): ", torch.isnan(part_queries).any())
                    # print("torch.isnan(class_name_embedding).any(): ", torch.isnan(class_name_embedding).any())
                    
                elif self.part_query_mode == 'clip_query':
                    hidden_states = outputs.hidden_states[-1]
                    
                    part_query_indices_batch = llm_input['part_query_indices_batch']
                    obj_part_queries_dict = self.get_seg_query(hidden_states, part_query_indices_batch, self.clip_emb_part_atten_mask)
                    obj_part_queries = obj_part_queries_dict['seg_query']
                    obj_part_query_atten_mask = obj_part_queries_dict['atten_mask']
                    num_part_queries = obj_part_queries.shape[1]
                    # squeeze the obj_part_queries to [bs * num_topk_queries*num_part_queries, C]
                    obj_part_queries = obj_part_queries.contiguous().view(bs*num_part_queries, -1)
                    part_queries = self.part_queries_back_mapper(obj_part_queries) # [bs, num_topk_objects, num_part_queries, 256]
                    part_queries = part_queries.contiguous().view(bs, num_part_queries, -1)

                else:
                    raise ValueError("part_query_mode must be learnable_query or clip_query")  
                
                # Part-level Decoder
                part_outputs, _ = self.part_predictor(multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, \
                                                    part_queries=part_queries, topk=topk, num_part_queries=num_part_queries, custom_part_categories_idx=custom_part_categories_idx,
                                                    part_query_attn_mask = obj_part_query_atten_mask)
            
        # Determine topk predictions
        test_topk = self.inference_topk(task, topk)
        topk_object, topk_part = self.hierarchical_topk(task, topk)
        # print("topk_object: {}, topk_part:{}, task:{}, topk:{}".format(topk_object, topk_part, task, topk))
        
        # Get hierarchical predictions
        outputs = self.hierarchical_topk_outputs(task, object_level_task, part_level_task, batch_name_list, object_outputs, part_outputs, test_topk, topk_object, topk_part, custom_object_categories_idx, custom_part_categories_idx)
        return outputs

    def hierarchical_topk_outputs(self, task, object_level_task, part_level_task, batch_name_list, object_outputs, part_outputs, test_topk, topk_object, topk_part, custom_object_categories_idx, custom_part_categories_idx):
        outputs = {}
        # print("object_outputs['pred_logits'].shape: ", object_outputs['pred_logits'].shape)
        batch_size = object_outputs['pred_logits'].shape[0]
        top_object_outputs_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk_object, 1)[1]
        # print("topk_object: ", topk_object)
        # print("top_object_outputs_indices.shape: ", top_object_outputs_indices)
        top_object_outputs = {}
        top_object_outputs['pred_logits'] = torch.gather(object_outputs['pred_logits'], 1, top_object_outputs_indices.unsqueeze(-1).repeat(1, 1, object_outputs['pred_logits'].shape[-1]))
        top_object_outputs['pred_masks'] = torch.gather(object_outputs['pred_masks'], 1, top_object_outputs_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, object_outputs['pred_masks'].shape[-2], object_outputs['pred_masks'].shape[-1]))
        top_object_outputs['pred_boxes'] = torch.gather(object_outputs['pred_boxes'], 1, top_object_outputs_indices.unsqueeze(-1).repeat(1, 1, object_outputs['pred_boxes'].shape[-1]))
        
        # print("top_object_outputs['pred_logits'].shape: ", top_object_outputs['pred_logits'].shape)
        # print("top_object_outputs['pred_masks'].shape: ", top_object_outputs['pred_masks'].shape)
        # print("top_object_outputs['pred_boxes'].shape: ", top_object_outputs['pred_boxes'].shape)
        # print("test_topk: ", test_topk)
        
        
        if task in self.object_level_datasets or part_outputs is None:
            test_topk = min(test_topk, topk_object)
            if part_outputs is None: # meaning at training stage1, no loss on part seg
                if task == 'custom':
                    all_logits = torch.full((batch_size, test_topk, len(batch_name_list)), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, custom_object_categories_idx] = top_object_outputs['pred_logits']
                else:
                    all_logits = torch.full((batch_size, test_topk, self.num_classes[task]), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, self.object_category_index_mapper[object_level_task]] = top_object_outputs['pred_logits']
                top_object_outputs['pred_logits'] = all_logits
                # print("top_object_outputs['pred_logits'].shape: ", top_object_outputs['pred_logits'].shape)
                # print("top_object_outputs['pred_masks'].shape: ", top_object_outputs['pred_masks'].shape)
                # print("top_object_outputs['pred_boxes'].shape: ", top_object_outputs['pred_boxes'].shape)
            return top_object_outputs
        
        # print("part_outputs['pred_logits'].shape: ", part_outputs['pred_logits'].shape)
        # print("part_outputs['pred_logits'].max(-1)[0].shape: ", part_outputs['pred_logits'].max(-1)[0].shape)
        # print("topk_part: ", topk_part)
        
        
        max_logits = part_outputs['pred_logits'].max(-1)[0]
        num_parts = max_logits.shape[1]
        topk_part = min(topk_part, num_parts)
        test_topk = topk_object+topk_part
        
        # print("topk_part: ", topk_part)
        
        top_part_outputs_indices = torch.topk(max_logits, topk_part, 1)[1]
        top_part_outputs = {}
        top_part_outputs['pred_logits'] = torch.gather(part_outputs['pred_logits'], 1, top_part_outputs_indices.unsqueeze(-1).repeat(1, 1, part_outputs['pred_logits'].shape[-1]))
        top_part_outputs['pred_masks'] = torch.gather(part_outputs['pred_masks'], 1, top_part_outputs_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, part_outputs['pred_masks'].shape[-2], object_outputs['pred_masks'].shape[-1]))
        top_part_outputs['pred_boxes'] = torch.gather(part_outputs['pred_boxes'], 1, top_part_outputs_indices.unsqueeze(-1).repeat(1, 1, part_outputs['pred_boxes'].shape[-1]))
        
        # print("top_part_outputs['pred_logits'].shape: ", top_part_outputs['pred_logits'].shape)
        
        if task in self.test_part_only_datasets:
            return top_part_outputs
        
        for key in top_object_outputs.keys():
            if key == 'pred_logits':
                if task == 'custom':
                    all_logits = torch.full((batch_size, test_topk, len(batch_name_list)), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, custom_object_categories_idx] = top_object_outputs['pred_logits']
                    all_logits[:, topk_object:, custom_part_categories_idx] = top_part_outputs['pred_logits']
                else:
                    all_logits = torch.full((batch_size, test_topk, self.num_classes[task]), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, self.object_category_index_mapper[object_level_task]] = top_object_outputs['pred_logits']
                    all_logits[:, topk_object:, self.part_category_index_mapper[part_level_task]] = top_part_outputs['pred_logits']
                    
                outputs[key] = all_logits
            else:
                # print("top_object_outputs[{}]: {}, top_part_outputs[{}]]: {}".format(key, top_object_outputs[key].shape, key, top_part_outputs[key].shape))
                outputs[key] = torch.cat([top_object_outputs[key], top_part_outputs[key]], dim=1)
        return outputs
    
    def hierarchical_topk(self, task, topk):
        if self.visualize:
            topk_object = topk
            topk_part = self.num_part_queries
        else:
            if task == 'seginw_House-Parts':
                topk_object = 0
                topk_part = 100
            elif 'seginw' in task and task not in ['seginw_Airplane-Parts', 'seginw_Bottles']:
                topk_object = 100
                topk_part = 0
            else:
                topk_object = self.test_object_inst_num[task]
                topk_part = self.test_part_inst_num[task]
        return topk_object, topk_part
    
    def inference_topk(self, task, topk):
        if 'seginw' in task and not self.visualize:
            test_topk = 100
        else:
            test_topk = self.test_topk_per_image[task] if not self.visualize else topk + self.num_part_queries
        return test_topk

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def embed_classes(self, dataset_cls_ids_dict, dataset_cls_indice_dict, dataset_name):
        class_name_ids = dataset_cls_ids_dict[dataset_name]
        cls_indice = dataset_cls_indice_dict[dataset_name]
        # print("len(class_name_ids): ", len(class_name_ids))
        # print("len(cls_indice): ",len(cls_indice))
        # print("class_name_ids: ", class_name_ids)
        # print("cls_indice: ", cls_indice)
        class_name_ids_device = class_name_ids.to(self.device)
        cls_indice_device = cls_indice.to(self.device)
        
        # num_class = cls_indice_device.unique_consecutive()
        # num_class = num_class[num_class >= 0]
        
        # class_name_ids_device_list = [class_name_ids_device[cls_indice_device == idx] for idx in num_class]
        # embedded_class_name = [self.llm_model.base_model.model.model.embed_tokens(id) for id in class_name_ids_device_list]
        # return embedded_class_name
        
        class_name_embeds = self.llm_token_embed_func(class_name_ids_device)
        # print("class_name_embeds.shape: ", class_name_embeds.shape)
        return class_name_embeds, cls_indice_device
    
    def prepare_multimodal_llm_input(self, dataset_name, obj_queries, multi_scale_features, topk_classes = None, extra=None, images_embed = None, batch_names = None):
        """_summary_

        Args:
            multi_scale_features: [num_levels, bs, C, H, W]
            obj_queries: [bs, topk_num_queries, C]
        """
        # print("Preparing multimodal llm input...")
        # print("multi_scale_features[0].shape:", multi_scale_features[0].shape)
        # print("top_k queries.shape:", obj_queries.shape)
        bs, topk_num_queries, C = obj_queries.shape
        
        device = obj_queries.device
        bs = multi_scale_features[0].shape[0]
        
        inpu_ids_template = self.input_ids_template[0] # assume only one round of conversation
        # split input ids template into chunks based on special tokens 
        chunks = []
        current_chunk = []
        for id in inpu_ids_template:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=device))
        
        # num_img_token_ph = 0 if self.use_vlm else 1
        num_img_token_ph = 1
        pixel_values = None
        if self.use_vlm:
            assert images_embed is not None, "images_vlm is None! "

        if self.part_query_mode == 'learnable_query':
            # tokenize the input
            image_token_indices = torch.where(inpu_ids_template == IMAGE_TOKEN_INDEX)[0]
            # cls_token_indices = torch.where(inpu_ids_template == GENERAL_CLASS_INDEX)[0]
            obj_query_indices = torch.where(inpu_ids_template == OBJ_QUERY_INDEX)[0]
            part_query_indices = torch.where(inpu_ids_template == OBJ_PART_QUERY_INDEX)[0]
            assert len(image_token_indices) == num_img_token_ph, "There should be only one image token in the input ids template"
            # assert len(cls_token_indices) == 1, "There should be only one class index as place holder in the input ids template"
            assert len(obj_query_indices) == topk_num_queries, "There should be two times of object query indices in the input ids template. len(obj_query_indices): {}; topk_num_queries: {} ".format(len(obj_query_indices), topk_num_queries)
            assert len(part_query_indices) == self.num_part_queries * topk_num_queries, "There should be self.num_part_queries * topk_num_queries part indices in the input ids template. len(part_query_indices): {}; self.num_part_queries * topk_num_queries: {} ".format(len(part_query_indices), self.num_part_queries * topk_num_queries)
                
            # image tokens
            if not self.use_vlm:
                dense_features = multi_scale_features[2] # [bs, C_in, H, W]
                mapped_dense_features = self.vision_projector(dense_features) # [bs, H*W, C_out]
            else:
                mapped_dense_features = images_embed
            # print("mapped_dense_features: ", mapped_dense_features.shape)
            
            # class name embeddings
            # class_name_embeddings = self.embed_classes(self.dataset_class_ids, self.dataset_class_ids_indices, dataset_name)
            # class_name_embeds, cls_indice_device = self.embed_classes(self.dataset_class_ids, self.dataset_class_ids_indices, dataset_name)
                
            # obj part queries 
            ## squeeze the input to (bs*num_objects, C)
            obj_queries = obj_queries.contiguous().view(bs * topk_num_queries, -1) # [bs*num_objects, 256]
            obj_queries_llm_input = self.obj_queries_projector(obj_queries) # [bs, topk_num_queries, 2048]
            ## reshape the input to (bs, num_objects, C)
            obj_queries_llm_input = obj_queries_llm_input.contiguous().view(bs, topk_num_queries, -1) # [bs, num_objects, 2048]
            part_queries_batched = self.part_queries_llm.weight.unsqueeze(0).unsqueeze(0).repeat(bs, topk_num_queries, 1, 1) # [bs, num_objects, num_parts, 2048]
            ## contatenate the object queries and part queries
            obj_queries_batched = obj_queries_llm_input.unsqueeze(2).repeat(1, 1, self.num_part_queries, 1) # [bs, num_objects, num_parts, 2048]
            obj_part_queries_batched = torch.cat([obj_queries_batched, part_queries_batched], dim=-1) # [bs, num_objects, num_parts, 2048*2]
            # print("obj_part_queries_batched.shape before mapping:", obj_part_queries_batched.shape)
            ## sequeeze the input to (bs*num_objects*num_parts, C)
            obj_part_queries_batched = obj_part_queries_batched.contiguous().view(bs * topk_num_queries * self.num_part_queries, -1) # [bs*num_objects*num_parts, 2048*2]
            obj_part_queries_batched = self.obj_part_queries_mapper(obj_part_queries_batched) # [bs, num_objects, num_parts, 2048]
            ## reshape the input to (bs, num_objects, num_parts, C)
            obj_part_queries_batched = obj_part_queries_batched.contiguous().view(bs, topk_num_queries, self.num_part_queries, -1) # [bs, num_objects, num_parts, 2048]
            # print("obj_part_queries_batched.shape after mapping:", obj_part_queries_batched.shape)
            
            new_input_embeds_list = []
            cls_embed_indices_list = []
            candidate_obj_query_indices_list = []
            output_obj_query_indices_list = []
            part_query_indices_list = []
            torch.set_printoptions(threshold=float('inf'))
            for bid in range(bs):
                img_tokens = mapped_dense_features[bid]
                obj_queries_per_batch = obj_queries_llm_input[bid]
                obj_part_queries_per_batch = obj_part_queries_batched[bid]
                
                cur_new_input_embeds = []

                cur_cls_embed_indices = []
                cur_candidate_obj_query_indices = []
                cur_output_obj_query_indices = []
                cur_part_query_indices = []
                
                cur_obj_queries_idx = 0
                cur_part_queries_idx = 0
                
                for chunk in chunks:
                    chunk_len = len(chunk)
                    if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX: # image token
                        # print(" img_tokens.shape:", img_tokens.shape)
                        cur_new_input_embeds.append(img_tokens)
                        zero_mask = torch.full((img_tokens.shape[0],), 0, device=device, dtype=inpu_ids_template.dtype)
                        cur_cls_embed_indices.append(zero_mask)
                        cur_candidate_obj_query_indices.append(zero_mask)
                        cur_output_obj_query_indices.append(zero_mask)
                        cur_part_query_indices.append(zero_mask)
                        
                    # elif chunk_len == 1 and chunk[0] == GENERAL_CLASS_INDEX: # class token
                    #     # for cls_idx in range(len(class_name_embeddings)):
                    #     #     cur_new_input_embeds.append(class_name_embeddings[cls_idx])
                            
                    #     #     zero_mask = torch.full((class_name_embeddings[cls_idx].shape[0],), 0, device=device, dtype=inpu_ids_template.dtype)
                    #     #     cls_idx_mask = torch.full((class_name_embeddings[cls_idx].shape[0],), cls_idx + 1, device=device, dtype=inpu_ids_template.dtype)
                            
                    #     #     cur_cls_embed_indices.append(cls_idx_mask)
                    #     #     cur_candidate_obj_query_indices.append(zero_mask)
                    #     #     cur_output_obj_query_indices.append(zero_mask)
                    #     #     cur_part_query_indices.append(zero_mask)
                            
                    #     #     # print(" class token shape:", class_name_embeddings[cls_idx].shape)
                        
                    #     cur_new_input_embeds.append(class_name_embeds)
                    #     cur_cls_embed_indices.append(cls_indice_device)
                    #     zero_mask = torch.full((cls_indice_device.shape[0],), 0, device=device, dtype=inpu_ids_template.dtype)
                    #     cur_candidate_obj_query_indices.append(zero_mask)
                    #     cur_output_obj_query_indices.append(zero_mask)
                    #     cur_part_query_indices.append(zero_mask)
                            
                    elif chunk_len == 1 and chunk[0] == OBJ_QUERY_INDEX: # object query token
                        obj_query_idx = cur_obj_queries_idx % topk_num_queries
                        cur_new_input_embeds.append(obj_queries_per_batch[obj_query_idx: obj_query_idx+1])
                        
                        zero_mask = torch.full((1,), 0, device=device, dtype=inpu_ids_template.dtype)
                        obj_query_idx_mask = torch.full((1,), 1, device=device, dtype=inpu_ids_template.dtype)
                        
                        cur_cls_embed_indices.append(zero_mask)
                        if cur_obj_queries_idx < topk_num_queries:
                            cur_candidate_obj_query_indices.append(obj_query_idx_mask)
                            cur_output_obj_query_indices.append(zero_mask)
                        else:
                            cur_candidate_obj_query_indices.append(zero_mask)
                            cur_output_obj_query_indices.append(obj_query_idx_mask)
                        cur_part_query_indices.append(zero_mask)
                        
                        cur_obj_queries_idx += 1
                        
                        # print(" obj query shape:", cur_new_input_embeds[-1].shape)
                        
                    elif chunk_len == 1 and chunk[0] == OBJ_PART_QUERY_INDEX:
                        obj_idx = cur_obj_queries_idx // self.num_part_queries
                        part_idx = cur_part_queries_idx % self.num_part_queries
                        cur_new_input_embeds.append(obj_part_queries_per_batch[obj_idx, part_idx:part_idx+1])
                        
                        zero_mask = torch.full((1,), 0, device=device, dtype=inpu_ids_template.dtype)
                        objpart_query_idx_mask = torch.full((1,), 1, device=device, dtype=inpu_ids_template.dtype)
                        cur_cls_embed_indices.append(zero_mask)
                        cur_candidate_obj_query_indices.append(zero_mask)
                        cur_output_obj_query_indices.append(zero_mask)
                        cur_part_query_indices.append(objpart_query_idx_mask)
                        
                        cur_part_queries_idx += 1
                        
                        # print(" obj part query shape:", cur_new_input_embeds[-1].shape)
                        
                    else:
                        cur_new_input_embeds.append(self.llm_token_embed_func(chunk))
                        zero_mask = torch.full((chunk_len,), 0, device=device, dtype=inpu_ids_template.dtype)
                        cur_cls_embed_indices.append(zero_mask)
                        cur_candidate_obj_query_indices.append(zero_mask)
                        cur_output_obj_query_indices.append(zero_mask)
                        cur_part_query_indices.append(zero_mask)
                        
                        # print(" texts' token shape:", cur_new_input_embeds[-1].shape)
                    # print("cur_cls_embed_indices: ", cur_cls_embed_indices)
                    # print("cur_candidate_obj_query_indices: ", cur_candidate_obj_query_indices)
                    # print("cur_output_obj_query_indices: ", cur_output_obj_query_indices)
                    # print("cur_part_query_indices: ", cur_part_query_indices)
                    
                        
                assert cur_obj_queries_idx == topk_num_queries, "object queries should be used up, cur_obj_queries_idx: {}, topk_num_queries: {}".format(cur_obj_queries_idx, topk_num_queries)
                assert cur_part_queries_idx == self.num_part_queries * topk_num_queries, "part queries should be used up, cur_part_queries_idx: {}, self.num_part_queries * topk_num_queries: {}".format(cur_part_queries_idx, self.num_part_queries * topk_num_queries)
                
                cur_new_input_embeds_tensor = torch.cat(cur_new_input_embeds, dim=0) # [L, C_llm]
                cur_cls_embed_indices_tensor = torch.cat(cur_cls_embed_indices, dim=0) # [L, C_llm]
                cur_candidate_obj_query_indices_tensor = torch.cat(cur_candidate_obj_query_indices, dim=0) # [L, C_llm]
                cur_output_obj_query_indices_tensor = torch.cat(cur_output_obj_query_indices, dim=0) # [L, C_llm]
                cur_part_query_indices_tensor = torch.cat(cur_part_query_indices, dim=0) # [L, C_llm]
                new_input_embeds_list.append(cur_new_input_embeds_tensor)
                cls_embed_indices_list.append(cur_cls_embed_indices_tensor)
                candidate_obj_query_indices_list.append(cur_candidate_obj_query_indices_tensor)
                output_obj_query_indices_list.append(cur_output_obj_query_indices_tensor)
                part_query_indices_list.append(cur_part_query_indices_tensor)
                
                # print("cur_new_input_embeds_tensor.shape: ", cur_new_input_embeds_tensor.shape)
                # print("class_name_embeddings.shape: ", class_name_embeddings[0].shape)
                # print("num of cls_embed_indices: ", (cur_cls_embed_indices_tensor!=0).sum())
                # print("num of candidate_obj_query_indices: ", cur_candidate_obj_query_indices_tensor.sum())
                # print("num of output_obj_query_indices: ", cur_output_obj_query_indices_tensor.sum())
                # print("num of part_query_indices: ", cur_part_query_indices_tensor.sum())
          
            # list to tensor 
            new_input_embeds_batch = torch.stack(new_input_embeds_list, dim=0) # [bs, L, C_llm]
            cls_embed_indices_batch = torch.stack(cls_embed_indices_list, dim=0) # [bs, L, C_llm]
            candidate_obj_query_indices_batch = torch.stack(candidate_obj_query_indices_list, dim=0) # [bs, L, C_llm]
            output_obj_query_indices_batch = torch.stack(output_obj_query_indices_list, dim=0) # [bs, L, C_llm]
            part_query_indices_batch = torch.stack(part_query_indices_list, dim=0) # [bs, L, C_llm]
            
            attention_mask = torch.ones((new_input_embeds_batch.shape[0], new_input_embeds_batch.shape[1]), device=device, dtype=torch.long)
                
            
            print("new_input_embeds_batch.shape: ", new_input_embeds_batch.shape)
            return {
                'new_input_embeds_batch': new_input_embeds_batch, 
                'attention_mask': attention_mask,
                'cls_embed_indices_batch': cls_embed_indices_batch,
                'candidate_obj_query_indices_batch': candidate_obj_query_indices_batch,
                'output_obj_query_indices_batch': output_obj_query_indices_batch,
                'part_query_indices_batch': part_query_indices_batch,
                "pixel_values": pixel_values
            }
          
        elif self.part_query_mode == 'clip_query':
            # if dataset_name not in self.data_classes_dict:
            #     candidate_names = batch_names  
            #     candidate_from = "batch_names"
            # else:
            #     candidate_names = self.data_classes_dict[dataset_name] 
            #     candidate_from = "self.data_classes_dict[dataset_name] "
            candidate_names = batch_names  
            candidate_from = "batch_names"
            # tokenize the input
            image_token_indices = torch.where(inpu_ids_template == IMAGE_TOKEN_INDEX)[0]
            part_query_indices = torch.where(inpu_ids_template == OBJ_PART_QUERY_INDEX)[0]
            assert len(image_token_indices) == num_img_token_ph, "There should be only one image token in the input ids template"
            assert len(part_query_indices) == topk_num_queries, "There should be self.num_part_queries * topk_num_queries part indices in the input ids template. len(part_query_indices): {};  topk_num_queries: {} ".format(len(part_query_indices), topk_num_queries)
            assert extra['class_embeddings'].shape[0] == len(candidate_names), "Task: {}. The number of class embeddings should be equal to the number of classes in the dataset. extra['class_embeddings'].shape[0]: {}; len(self.data_classes_dict[dataset_name]): {}".format(dataset_name, extra['class_embeddings'].shape[0], len(candidate_names))
            
            if dataset_name in self.cat_hierachies:
                cat_hierarchy = self.cat_hierachies[dataset_name]
            else:
                cat_hierarchy = {'id': {},}
            
            # image tokens
            if not self.use_vlm:
                dense_features = multi_scale_features[2] # [bs, C_in, H, W]
                mapped_dense_features = self.vision_projector(dense_features) # [bs, H*W, C_out]
            else:
                mapped_dense_features = images_embed

            # print("mapped_dense_features: ", mapped_dense_features.shape)
            
            # prepare obj part queries - clip embeddings
            obj_part_query_list = []
            obj_part_query_indices_list = []
            part_ids_list = []
            for bid in range(bs):
                obj_part_query_per_batch_indices_list = []
                concat_query_idx = 0
                part_ids_per_batch_list = []
        
                obj_queries_per_batch = obj_queries[bid] # [topk_num_queries, 256]
                obj_classes_per_batch = topk_classes[bid]
                obj_part_concat_query_per_batch_list = []
                for qid in range(topk_num_queries):
                    obj_class_per_query = obj_classes_per_batch[qid].item()
                    if obj_class_per_query in cat_hierarchy['id']:
                        # assert obj_class_per_query in cat_hierarchy['id'], "obj_class_per_query: {} not in cat_hierarchy {}".format(obj_class_per_query, cat_hierarchy)
                        obj_part_ids = cat_hierarchy['id'][obj_class_per_query]
                        part_embeddings = extra['class_embeddings'][obj_part_ids] # [num_part_queries_per_obj, 256]
                        
                        obj_query = obj_queries_per_batch[qid].unsqueeze(0).repeat(len(obj_part_ids), 1) # [num_part_queries_per_obj, 256]
                        obj_part_concat_query = torch.concat([obj_query, part_embeddings], dim=-1) # [num_part_queries_per_obj, 256+256]
                        
                        obj_part_concat_query_per_batch_list.append(obj_part_concat_query)
                        obj_part_query_per_batch_indices_list.append([concat_query_idx + idx for idx in range(len(obj_part_ids))])
                        concat_query_idx += len(obj_part_ids)
                        part_ids_per_batch_list.append(obj_part_ids)
                    else:
                        obj_part_concat_query_per_batch_list.append( torch.zeros((1, self.obj_part_concat_dim), device = device) )
                        obj_part_query_per_batch_indices_list.append([concat_query_idx])
                        concat_query_idx += 1
                        part_ids_per_batch_list.append([-1])
                        
                
                obj_part_concat_query_per_batch = torch.cat(obj_part_concat_query_per_batch_list, dim=0) # [num_concat_part_queries, 256+256]
                obj_part_query_per_batch = self.obj_part_queries_mapper(obj_part_concat_query_per_batch) # [num_concat_part_queries, 2048]
                obj_part_query_list.append(obj_part_query_per_batch)
                obj_part_query_indices_list.append(obj_part_query_per_batch_indices_list)
                part_ids_list.append(part_ids_per_batch_list)
                
            # prepare input embeds
            new_input_embeds_list = []

            output_obj_query_indices_list = []
            part_query_indices_list = []
            torch.set_printoptions(threshold=float('inf'))
            for bid in range(bs):
                img_tokens = mapped_dense_features[bid]

                obj_part_query_per_batch = obj_part_query_list[bid]
                obj_part_query_per_batch_indices = obj_part_query_indices_list[bid]
                obj_idx = 0
                
                cur_new_input_embeds = []
                cur_candidate_obj_query_indices = []
                # print("getting chunk embs for batch ", bid)
                for chunk in chunks:
                    chunk_len = len(chunk)
                    if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX: # image token
                        # print(" img_tokens.shape:", img_tokens.shape)
                        cur_new_input_embeds.append(img_tokens)
                        
                        zero_mask = torch.full((img_tokens.shape[0],), 0, device=device, dtype=inpu_ids_template.dtype)
                        cur_candidate_obj_query_indices.append(zero_mask)
                    elif chunk_len == 1 and chunk[0] == OBJ_PART_QUERY_INDEX:
                        
                        obj_part_query_per_obj_indices = obj_part_query_per_batch_indices[obj_idx]
                        obj_part_query_per_obj = obj_part_query_per_batch[obj_part_query_per_obj_indices]
                        cur_new_input_embeds.append(obj_part_query_per_obj)
                        # print(" obj_part_query_per_obj.shape:", obj_part_query_per_obj.shape)
                        
                        obj_part_query_per_obj_mask = torch.full((len(obj_part_query_per_obj_indices),), obj_idx + 1, device=device, dtype=inpu_ids_template.dtype)
                        # zero_mask = torch.full((len(obj_part_query_per_obj_indices),), 0, device=device, dtype=inpu_ids_template.dtype)
                        cur_candidate_obj_query_indices.append(obj_part_query_per_obj_mask)
                        obj_idx += 1
                    else:
                        cur_new_input_embeds.append(self.llm_token_embed_func(chunk))
                        # print(" chunk.shape:", cur_new_input_embeds[-1].shape)
                        zero_mask = torch.full((chunk_len,), 0, device=device, dtype=inpu_ids_template.dtype)
                        cur_candidate_obj_query_indices.append(zero_mask)
                        
                cur_new_input_embeds_tensor = torch.cat(cur_new_input_embeds, dim=0) # [L, C_llm]
                cur_candidate_obj_query_indices_tensor = torch.cat(cur_candidate_obj_query_indices, dim=0) # [L, C_llm]
                
                new_input_embeds_list.append(cur_new_input_embeds_tensor)
                part_query_indices_list.append(cur_candidate_obj_query_indices_tensor)
                
            # pad new_input_embeds_list to tensor and set attention_mask
            max_token_length = max([len(x) for x in new_input_embeds_list])
            new_input_embeds_batch = torch.nn.utils.rnn.pad_sequence(
                    new_input_embeds_list,
                    batch_first=True,
                    padding_value=-1,
                )
            attention_mask = torch.zeros((len(new_input_embeds_list), max_token_length), device=device, dtype=torch.long)
            for i, x in enumerate(new_input_embeds_list):
                attention_mask[i, :len(x)] = 1
            assert attention_mask.shape == new_input_embeds_batch.shape[:2], "Attention mask shape should be same as input embeds batch shape. attention_mask.shape: {}; new_input_embeds_batch.shape: {}".format(attention_mask.shape, new_input_embeds_batch.shape)

            part_query_indices_batch = torch.nn.utils.rnn.pad_sequence(
                    part_query_indices_list,
                    batch_first=True,
                    padding_value=0,
                )
            print("new_input_embeds_batch.shape: ", new_input_embeds_batch.shape)
            return {
                'new_input_embeds_batch': new_input_embeds_batch, 
                'part_query_indices_batch': part_query_indices_batch,
                'attention_mask': attention_mask, 
                'part_ids_list': part_ids_list, 
                "pixel_values": pixel_values
            }
            
        else:
            raise NotImplementedError("part_query_mode {} is not supported".format(self.part_query_mode))
        
    
    def get_class_name_embedding(self, hidden_states, cls_token_indices):
        class_name_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, cls_token_indices):
            class_id = torch.unique(current_token_indice)
            class_id = class_id[class_id != 0]
            current_class_name_embedding_list = []
            for id in class_id:
                current_class_mask = (current_token_indice == id)
                current_class_state = current_hidden_state[current_class_mask]
                current_class_name_embedding_list.append(current_class_state)
            current_pool_class_name_embedding = [self.class_name_pooling(class_name.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous()
                                                 for class_name in current_class_name_embedding_list]
            # for i in range(len(current_pool_class_name_embedding)):
            #     print("current_pool_class_name_embedding.shape:", current_pool_class_name_embedding[i].shape)
            class_name_embedding_list.append(torch.cat(current_pool_class_name_embedding, dim=0))
        return torch.stack(class_name_embedding_list, dim=0)
    
    def get_seg_query(self, hidden_states, seg_query_masks, get_atten_mask=False):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]
            seg_query_per_batch_list = []
            for value in unique_query_value:
                current_query_mask = (sample_query_mask == value)
                current_query = sample_hidden_state[current_query_mask]
                seg_query_per_batch_list.append(current_query)
            seg_query_per_batch = torch.cat(seg_query_per_batch_list, dim=0)
            # print("seg_query_per_batch.shape: ", seg_query_per_batch.shape)
            seg_query_list.append(seg_query_per_batch)

        # if same size, stack, otherwise keep it as list
        ## TODO: check if all seg_query_list are same size
        
        is_same_size = all([x.shape == seg_query_list[0].shape for x in seg_query_list])
        if is_same_size:
            seg_query = torch.stack(seg_query_list, dim=0)
        else:
            # pad the list to same size
            seg_query = torch.nn.utils.rnn.pad_sequence(
                seg_query_list,
                batch_first=True,
                padding_value=0,
            )
            
        if get_atten_mask:
            if is_same_size:
                # part_query_atten_mask = torch.zeros((len(seg_query_list), seg_query.shape[1]), device=hidden_states.device, dtype=torch.bool)
                part_query_atten_mask = None
            else:
                # part_query_atten_mask = torch.zeros((len(seg_query_list), seg_query.shape[1]), device=hidden_states.device, dtype=torch.bool)
                # instead of using real atten_mask, we can just mark the start pad idx 
                part_query_atten_mask = []
                for bid in range(len(seg_query_list)):
                    part_query_atten_mask.append( seg_query_list[bid].shape[0] )
            
            # print("seg_query.shape: ", seg_query.shape)
            return {
                'seg_query': seg_query,
                'atten_mask': part_query_atten_mask
            }
                 
        else:
            return {
                'seg_query': seg_query,
                'atten_mask': None
            }
                    