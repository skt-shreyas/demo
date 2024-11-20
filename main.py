from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pathlib import Path
import os
from pydantic import BaseModel
from pandas import read_csv
app = FastAPI()
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import torchvision.transforms as transforms
import clip 
import torch
import os
import itertools
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from models import RED_DOT #_dom_vec_attn
from sklearn import metrics
# from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity
import clip
from utils import (
    set_seed,
    prepare_input
)
import zipfile
import shutil
import ast
transform = transforms.Compose([transforms.PILToTensor()])
# choose_gpu=0
# device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")
# print(device)

device = "cpu" #cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
clip_device = "cpu" #"cuda"

# base_clip, preprocess = clip.load("ViT-L/14", device=device)
# clip_model = base_clip

k_fold = 1
num_evidence = 10
RED_DOT_version = "baseline"
use_evidence=num_evidence   #10
use_evidence_neg=0
k_fold=k_fold
choose_fusion_method = [
                    ["concat_1", "add", "sub", "mul"]
                                        ]

encoder = 'CLIP'
encoder_version = 'ViT-L/14' #chang
epochs=100
seed_options = [0]
lr_options = [1e-4]
batch_size_options = [64]
tf_layers_options = [4]
tf_head_options = [8]
tf_dim_options = [128]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
init_model_name = '_RED_DOT_' + str(use_evidence) + '_' + RED_DOT_version 
results_filename = "results"
token_level = False
fuse_evidence_options = [["concat_1"]] if use_evidence else [[False]]
num_workers=8
early_stop_epochs = 10

choose_encoder_version = "ViT-L/14"

seed = 0
set_seed(seed_options[0])
lr = lr_options[0]
batch_size = batch_size_options[0]
tf_layers = tf_layers_options[0]
tf_head = tf_head_options[0]
tf_dim = tf_dim_options[0]
fusion_method = choose_fusion_method[0]
fuse_evidence = fuse_evidence_options[0]
fold = 1

torch.manual_seed(seed)

parameters = {
    "LEARNING_RATE": lr,
    "EPOCHS": epochs, 
    "BATCH_SIZE": batch_size,
    "TF_LAYERS": tf_layers,
    "TF_HEAD": tf_head,
    "TF_DIM": tf_dim,
    "NUM_WORKERS": 8,
    "USE_FEATURES": ["images", "texts"],
    "EARLY_STOP_EPOCHS": early_stop_epochs,
    # "CHOOSE_DATASET": dataset_name,
    "ENCODER": encoder,
    "ENCODER_VERSION": "ViT-L/14", #change
    "SEED": seed,
    "FUSION_METHOD": fusion_method, 
    "NETWORK_VERSION": RED_DOT_version,
    "TOKEN_LEVEL": token_level,
    "USE_EVIDENCE": use_evidence,
    "USE_NEG_EVIDENCE": use_evidence_neg,
    "FUSE_EVIDENCE": fuse_evidence,
    "k_fold": k_fold,
    "current_fold": fold                 
}


if parameters["ENCODER_VERSION"] == 'ViT-B/32':
    emb_dim_ = 512

elif parameters["ENCODER_VERSION"] == 'ViT-L/14':    
    emb_dim_ = 768

parameters["EMB_SIZE"] = emb_dim_             
model = RED_DOT(
    tf_layers=parameters["TF_LAYERS"],
    tf_head=parameters["TF_HEAD"],
    tf_dim=parameters["TF_DIM"],
    emb_dim=parameters["EMB_SIZE"],
    skip_tokens=len(fusion_method) if "concat_1" not in fusion_method else len(fusion_method) + 1,
    use_evidence=parameters["USE_EVIDENCE"],
    use_neg_evidence=parameters["USE_NEG_EVIDENCE"],
    model_version = RED_DOT_version,
    device=device,
    fuse_evidence=fuse_evidence,
)

model.to(device)
criterion = nn.BCEWithLogitsLoss()
criterion_mlb = nn.BCEWithLogitsLoss()                                                               

optimizer = torch.optim.Adam(
    model.parameters(), lr=parameters["LEARNING_RATE"]
)


# PATH = "./models/reddot_l14_baseline.pt"  
PATH = './models/17thSep_model(4-8-128)_news_clippings_balanced_multimodal_0_RED_DOT_1_baseline.pt'

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]

lavis_clip_model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", 
                                                                        model_type="ViT-L-14", 
                                                                        is_eval=True, 
                                                                        device=clip_device)   
lavis_clip_model.to(clip_device)
lavis_clip_model.eval()

# import sys
# import clip_B32 as clip_base

# sys.path.append('./DPOD-main')
# import clip_classifier as clip_classifier

# exp_folder = "./DPOD-main/model_saved_path"
# best_model_name = 'best_model_acc.pth.tar'

# model_settings = {'pdrop': 0.00}
# base_clip_B32, preprocess_base = clip_base.load("ViT-B/32", device=clip_device, jit=False)
# classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip_B32)
# classifier_clip.to(clip_device)
# criterion = nn.BCEWithLogitsLoss(reduction='none').to(clip_device)

# checkpoint = torch.load(os.path.join(exp_folder, best_model_name), map_location=device)
# classifier_clip.load_state_dict(checkpoint['state_dict'], strict=False)

# clip_base.model_B32.convert_weights(classifier_clip)
# classifier_clip.eval()
# print("=> loaded checkpoint: '{}')".format(os.path.join(exp_folder, 'best_model_acc.pth.tar')))
# print("ACLIP version loaded")

from transformers import LongformerTokenizer, LongformerModel, ViTFeatureExtractor, ViTModel, BertTokenizer, BertModel
from skimage.transform import resize 
from skimage import io as io
import torch, pickle

from SAFE.SAFE_utils import *

import torch.nn.functional as F
from torch import nn

# text_model_SAFE = LongformerModel.from_pretrained('allenai/longformer-base-4096')
# tokenizer_SAFE = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# feature_extractor_SAFE = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# vision_model_SAFE = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# net_SAFE = Siamese(latent_dim=256).to("cuda")
# net_SAFE.load_state_dict(torch.load('./SAFE/Gossip_best_model.pth')['model_state_dict'])
# net_SAFE.eval()
# classifier_SAFE = pickle.load(open("./SAFE/classifier_svm.sav", 'rb'))

checkpoint_safe = torch.load('./models/test11_with_Clip_model_nc.pth.tar')
# model_safe=Siamese(latent_dim=256).to(device)
# model_safe.load_state_dict(checkpoint_safe['model_state_dict'])
# model_safe.eval()
mlp_model = MLP().to(device)
mlp_model.load_state_dict(checkpoint_safe['mlp_model_state_dict'])
mlp_model.eval()

print('SAFE Classifier loaded')

def get_clip_feature_queries(img_path, caption):
    with torch.no_grad():
        image = Image.open(img_path)
        image = image.convert('RGB')
        max_size = 400
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height))
        img = vis_processors["eval"](image).unsqueeze(0)        
        txt = txt_processors["eval"](caption)
        
        sample = {
                        "image": img.to(clip_device),
                        "text_input": txt
        }

        clip_features = lavis_clip_model.extract_features(sample)                

        image_features = clip_features.image_embeds_proj
        text_features = clip_features.text_embeds_proj

        ##
        text_features = text_features.reshape(-1)
        text_features = text_features.detach()

        # image_features = image_features.image_embeds_proj
        image_features = image_features.reshape(-1).detach()#.numpy()
    return image_features, text_features

def get_clip_img_feature(img_path):
    with torch.no_grad():
        image = Image.open(img_path)
        image = image.convert('RGB')
        max_size = 400
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height))
        img = vis_processors["eval"](image).unsqueeze(0)        
        sample = {
                        "image": img.to(clip_device)
        }
        image_features = lavis_clip_model.extract_features(sample)                
        image_features = image_features.reshape(-1).detach()#.numpy()
    return image_features

def get_clip_text_feature(caption):
    with torch.no_grad():
        txt = txt_processors["eval"](caption)
        sample = {
                        "text_input": txt
        }
        clip_features = lavis_clip_model.extract_features(sample)
        text_features = clip_features
        text_features = text_features.reshape(-1)
        text_features = text_features.detach()#.numpy()
    return text_features

def unzip_file(zip_path, extract_to):
    """
    Extracts a zip file to a specified location.

    Args:
    zip_path (str): Path to the zip file.
    extract_to (str): Destination folder where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)

def run_all_algos(processed_items, q_img, q_caption, evidence_images_path, evidence_captions, entities, req_list=False):

    SAFE_result = None
    SAFE_probabilities = None
    prediction_ACLIP = None
    image_attention = None
    predictions = None
    CCN_result = None
    top_caption_indices = None
    top_attended_images_names = None

    # if processed_items[0] == 'TRUE':
    #     # image_SAFE = resize(io.imread(q_img , plugin = 'matplotlib')/255, (224,224,3), anti_aliasing=True)
    #     # vision_features_SAFE = vision_model_SAFE(**feature_extractor_SAFE(image_SAFE,return_tensors='pt')).pooler_output[0].detach()
    #     # text_features_SAFE = text_model_SAFE(**tokenizer_SAFE(q_caption_orig, return_tensors= 'pt')).pooler_output[0].detach()#.numpy()[0]
    #     # img_text_ft_SAFE = torch.concatenate((text_features_SAFE, vision_features_SAFE)).reshape(1,-1)

    #     # t_SAFE,v_SAFE = net_SAFE.forward_one(img_text_ft_SAFE.reshape(1,-1))

    #     # combined_ft_SAFE = torch.concatenate((t_SAFE,v_SAFE), axis = 1).cpu().detach().numpy()
    #     # SAFE_result = classifier_SAFE.predict(combined_ft_SAFE)
    #     # SAFE_probabilities = classifier_SAFE.predict_proba(combined_ft_SAFE)
    #     q_img_safe, q_caption_safe = get_clip_feature_queries(q_img, q_caption)
    #     text_tensor = torch.tensor(q_caption_safe).float().unsqueeze(0).to(device)  # Add batch dimension
    #     image_tensor = torch.tensor(q_img_safe).float().unsqueeze(0).to(device)
    #     img_text_features = np.append(text_tensor.cpu().numpy(), image_tensor.cpu().numpy(), axis=1)
    #     with torch.no_grad():
    #         #t_SAFE, v_SAFE = model.forward_one(torch.from_numpy(img_text_features))
    #         # breakpoint()
    #         #combined_features = torch.cat((t_SAFE, v_SAFE), dim=1)
    #         # y_pred = mlp_model(combined_features)
    #         SAFE_result = mlp_model(torch.tensor(img_text_features).to('cuda'))
        # breakpoint()
        
    # if processed_items[1] == 'TRUE':

    #     q_img_A = Image.open(q_img)
    #     singletext_tokenized = clip_base.tokenize(q_caption,truncate=True)
    #     q_img_A = preprocess_base(q_img_A)
    #     q_img_A = q_img_A.unsqueeze(0)
    #     with torch.no_grad():
    #         image_attention = classifier_clip.clip.encode_image_attention(q_img_A.to(device))
    #     q_img_A = q_img_A.to(clip_device)

    #     text_token = singletext_tokenized.to(clip_device)
    #     text_token = text_token.to(torch.int32)

    #     with open("./uploaded_texts/dom_embed.txt") as file:
    #         for line in file:
    #             line = line.strip()
    #             break
    #     domain_vec = line.split(',')
    #     domain_vec = [float(num) for num in domain_vec]
    #     domain_vec = [domain_vec]

    #     with torch.no_grad():
    #         prediction_ACLIP = classifier_clip(q_img_A, text_token, [q_caption], domain_vec).detach()

    # if processed_items[2] == 'TRUE':
    #     CCN_result = CCN_inference(q_img, q_caption, evidence_images_path, evidence_captions, entities)

    if processed_items[3] == 'TRUE':
        q_img, q_caption = get_clip_feature_queries(q_img, q_caption)

        try:
            evidence_images = [os.path.join("./uploaded_images/",imgfile) for imgfile in os.listdir(evidence_images_path)]
            if req_list == True:
                evidence_images = [os.path.join(evidence_images_path, file) for file in os.listdir(evidence_images_path)]
        except:
            pass

        X_img = [get_clip_img_feature(img_path) for img_path in evidence_images]
        X_img = torch.stack(X_img).to(device)
        X_txt = [get_clip_text_feature(txt) for txt in evidence_captions]
        X_txt = torch.stack(X_txt).to(device)

        cos_sim = cosine_similarity(q_img.to(device).reshape(1, -1), X_img) # Calculate the cosine similarity
        image_evidences_ranks = torch.argsort(-cos_sim).tolist()
        X_img = X_img[image_evidences_ranks]
        

        cos_sim = cosine_similarity(q_caption.to(device).reshape(1, -1), X_txt) # Calculate the cosine similarity
        text_evidences_ranks = torch.argsort(-cos_sim).tolist()
        X_txt = X_txt[text_evidences_ranks]

        use_evidence = 10


        if X_img.shape[0] < use_evidence:            
            pad_zeros = torch.zeros((use_evidence - X_img.shape[0], X_img.shape[1]), device=device)
            X_img = torch.vstack([X_img, pad_zeros])

        if X_txt.shape[0] < use_evidence:
            pad_zeros = torch.zeros((use_evidence - X_txt.shape[0], X_txt.shape[1]), device=device)
            X_txt = torch.vstack([X_txt, pad_zeros])          

        X_all = torch.concatenate([X_img, X_txt]).to("cpu") #("cuda")
        X_all_labels = torch.ones(X_img.shape[0] + X_txt.shape[0]).to("cpu") #("cuda")
        X_all = X_all.to(torch.float32)

        y_true = []
        y_pred = []
        y_pred_X_labels = []
        y_true_X_labels = []
        model.eval()

        images = q_img.reshape(1,-1).to(device, non_blocking=True)          # 768
        texts = q_caption.reshape(1,-1).to(device, non_blocking=True)           # 768                   
        labels = torch.tensor([1,1]).to(device, non_blocking=True)          # 1
        
        X_all = X_all.reshape(1,-1,768).to(device, non_blocking=True)           # 20 x 768 

        with open('saved_features.txt', 'w') as fp:
            for item in X_all.tolist():
                fp.write(f"{item}\n")

        with open('saved_qimg_features.txt', 'w') as fp:
            for item in images.tolist():
                fp.write(f"{item}\n")

        with open('saved_qtext_features.txt', 'w') as fp:
            for item in texts.tolist():
                fp.write(f"{item}\n")
        

        X_labels = X_all_labels.to(device, non_blocking=True)        # 20
        caption = " "        
        # breakpoint()

        x = prepare_input(fusion_method, fuse_evidence, use_evidence, images, texts, None, None, X_all)
        predictions = model(x, False, X_labels)
        all_head_attn_maps = predictions[2]
        stacked_attns= torch.stack(all_head_attn_maps)
        attn_map = torch.mean(stacked_attns, dim=0) #average across all 6 attn heads
        
        num_text_evidences = len(evidence_captions)
        num_img_evidences = len(evidence_images)

        attn_text_evidences_text = attn_map.squeeze(0)[0, 16: 16+num_text_evidences] #cls_tken attns over the text evidences
        top_values_text, top_indices_text = torch.topk(attn_text_evidences_text, k=min(num_text_evidences, 3))

        attn_img_evidences_images = attn_map.squeeze(0)[0, 6: 6+num_img_evidences] #cls_tken attns over the image evidences
        top_values_img, top_indices_img = torch.topk(attn_img_evidences_images, k=min(num_img_evidences, 3))

        evidence_images_names = [img_path.split('/')[-1] for img_path in evidence_images]       # Order in which images are loaded
        X_all_ordered_images = [evidence_images_names[i] for i in image_evidences_ranks]

        top_attended_images_names = [X_all_ordered_images[i] for i in top_indices_img.tolist()]
        top_caption_indices = [text_evidences_ranks[i] for i in top_indices_text]

    # if SAFE_result is not None:
    #     SAFE_result = SAFE_result[0].item()
    # if SAFE_probabilities is not None:
    #     SAFE_true_prob = SAFE_probabilities[0,0].item()
    #     SAFE_fake_prob = SAFE_probabilities[0,1].item()
    # if CCN_result is not None:
    #     CCN_result = CCN_result.item()
    # if prediction_ACLIP is not None:
    #     prediction_ACLIP  = prediction_ACLIP.detach().item()
    # if image_attention is not None:
    #     image_attention = image_attention.detach().cpu().tolist()
    if predictions is not None:
        predictions = predictions[0].detach().item()
    # print('SAFE Result and Probability : ', SAFE_result, SAFE_probabilities)
    # print("ACLIP Output", prediction_ACLIP)
    # print("Image Attention", image_attention)
    print("REDDOT Prediction", predictions)
    print("REDDOT : Top text indices", top_caption_indices)
    print("REDDOT : Top image names", top_attended_images_names)
    print("CCN_result", CCN_result)
    response_list = [SAFE_result, SAFE_probabilities, prediction_ACLIP, image_attention, predictions, CCN_result,top_caption_indices,top_attended_images_names]
    # response_list = [predictions,top_caption_indices,top_attended_images_names]

    # breakpoint()
    response = {"processed_items": response_list}
    return response

@app.post("/upload_data/")
async def upload_data(files: list[UploadFile], captions: list[str], entities: list[str]):
    try:
        # Create a folder to store uploaded images (if it doesn't exist)
        image_folder = Path("uploaded_images")
        image_folder.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(files):
         
            image_path = image_folder / file.filename
            with image_path.open("wb") as image_file:
                image_file.write(file.file.read())

        text_folder = Path("uploaded_texts")
        with open(os.path.join("./uploaded_texts", "captions.txt"), "w") as outfile:
            outfile.writelines(f"{s}\n" for s in captions)

        with open(os.path.join("./uploaded_texts", "entities.txt"), "w") as outfile:
            outfile.writelines(f"{s}\n" for s in entities)

        return JSONResponse(content={"message": "Images successfully uploaded and saved."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

import sys, copy
# sys.path.append('./CCN/sent_emb')
# #from ..CCN.sent_emb import get_inference_CCN
# from get_inference_CCN import *

@app.post("/upload_queries/")
async def upload_queries(files: list[UploadFile], captions: list[str]):
    try:
        image_folder = Path("uploaded_texts")
        image_folder.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(files):
            file.filename = "qimg.jpg"
            image_path = image_folder / file.filename
            with image_path.open("wb") as image_file:
                image_file.write(file.file.read())
        with open(os.path.join("./uploaded_texts", "query_cap.txt"), "w") as outfile:
            outfile.writelines(f"{s}\n" for s in captions)
        return JSONResponse(content={"message": "Images successfully uploaded and saved."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload_zip_file/")
async def upload_zip_file(files: list[UploadFile], captions: list[str]):
    try:
        zip_folder = Path("uploaded_zip_file")
        zip_folder.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(files):
            file.filename = "all_files.zip"
            zip_path = zip_folder / file.filename
            with zip_path.open("wb") as zip_file:
                zip_file.write(file.file.read())
        
        return JSONResponse(content={"message": "zip file successfully uploaded and saved."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

class ItemList(BaseModel):
    items: list[str]

@app.post("/process_all_items")
async def process_list(item_list: ItemList):
    dest_path = "/home/suraj/shwetabh_files/all_algo_with_gui/uploaded_list_data"
    os.makedirs(dest_path, exist_ok = True)
    zip_file_path = '/home/suraj/shwetabh_files/all_algo_with_gui/uploaded_zip_file/all_files.zip'
    unzip_file(zip_file_path, dest_path)
    processed_items = [item.upper() for item in item_list.items]
    results = get_results_for_all_list(processed_items)
    return results

@app.post("/clear_all_list_items")
async def clear_all_list_items():
    folder_dest_path = "/home/suraj/shwetabh_files/all_algo_with_gui/uploaded_list_data"
    zip_folder_path = '/home/suraj/shwetabh_files/all_algo_with_gui/uploaded_zip_file'
    if os.path.exists(folder_dest_path):
        shutil.rmtree(folder_dest_path)
    if os.path.exists(zip_folder_path):
        shutil.rmtree(zip_folder_path)

def get_results_for_all_list(processed_items):
    base_folder_path = '/home/suraj/shwetabh_files/all_algo_with_gui/uploaded_list_data'
    results = {}
    for sample in os.listdir(base_folder_path):
        sample_path = os.path.join(base_folder_path, sample)
        entities = ['']
        if os.path.isdir(sample_path):
            for file in os.listdir(sample_path):
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    q_img = os.path.join(sample_path, file)
                if file == 'caption.txt':
                    caption_file_path = os.path.join(sample_path, file)
                    with open(caption_file_path) as cur_file:
                        q_caption = [line.strip() for line in cur_file][0]
                if file == 'evi_captions.txt':
                    caption_file_path = os.path.join(sample_path, file)
                    with open(caption_file_path) as cur_file:
                        evidence_captions = [line.strip() for line in cur_file]
                if file == 'evidences': 
                    evidence_images_path = os.path.join(sample_path, file)
                if file == 'entities.txt':
                    entities_file_path = os.path.join(sample_path, file)
                    try:
                        with open(entities_file_path) as cur_file:
                            entities = [line.strip() for line in cur_file]
                            entities = ast.literal_eval(entities[0])
                    except:
                        entities = ['']
            response = run_all_algos(processed_items, q_img, q_caption, evidence_images_path, evidence_captions, entities, req_list = True)
            results[sample] = response

    return results
@app.post("/upload_algo_request/")
async def process_items(item_list: ItemList):
    # Process the list of strings
    processed_items = [item.upper() for item in item_list.items]
    # breakpoint()
    try:
        with open("./uploaded_texts/captions.txt") as file:
            evidence_captions = [line.strip() for line in file]
    except:
        evidence_captions = []
        pass
     
    with open("./uploaded_texts/query_cap.txt") as file:
        q_caption = [line.strip() for line in file][0]
        q_caption_orig = copy.deepcopy(q_caption)

    try:
        with open("./uploaded_texts/entities.txt") as file:
            entities = [line.strip() for line in file]
    except:
        entities = ['']

    clip_device = 'cpu' #'cuda'
    device = 'cpu' #"cuda"
    q_img = "./uploaded_texts/qimg.jpg"
    q_image_path = "./uploaded_texts/qimg.jpg"
    evidence_images_path = "./uploaded_images/"
    # Create a JSON response
    response = run_all_algos(processed_items, q_img, q_caption, evidence_images_path, evidence_captions, entities)
    return response

@app.post("/list_images/")
def listImages():
    try:
        for files in os.listdir("./uploaded_images"):
            print(files)
        print("And Captions are:")
        for texts in captions_all:
            print(texts)        
    except:
        print("Some error occured")
    return "Printed in server"


@app.post("/clear_uploads/")
def clearuploads():
    try:
        filelist = [f for f in os.listdir("./uploaded_images/")]
        for f in filelist:
            os.remove(os.path.join("./uploaded_images/", f))
        os.remove(os.path.join("./uploaded_texts/","captions.txt"))
        os.remove(os.path.join("./uploaded_texts/","entities.txt"))
    except:
        return JSONResponse(content={"error": "some error occured while clearing the uploads"}, status_code=500)

@app.post("/save_dom_embed")
async def save_dom_embed(numbers: list[float]):
    with open("./uploaded_texts/dom_embed.txt", "w") as file:
        file.write(",".join(map(str, numbers)))
    return {"message": "List saved successfully"}
