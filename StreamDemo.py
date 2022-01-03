import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
from io import BytesIO
import base64
import numpy as np
import shutil
import matplotlib.pyplot as plt
#Create sample a app
st.title("Face Recognition Demo")
try:
    uploaded_file = st.file_uploader("Upload a file image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image uploaded', use_column_width=True, clamp=True)
except Exception as e:
    st.write("Can't Show Image,  Please Select Image Other To Show!")

def predict_face(img):
    # initializing MTCNN and InceptionResnetV1 
    mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    data_peoples = []
    min_dist = 0
    name_peoples = []
    box_peoples = []
    min_dist_list = []
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
        count_Unknown = 0
        if boxes is not None:
            # for box in boxes:
            for i, prob in enumerate(prob_list):
                if prob > 0.90:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                    dist_list = []  # list of matched distances, minimum distance is used to identify the person
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)
                        min_dist = min(dist_list)  # get minumum dist value
                        min_dist_idx = dist_list.index(
                            min_dist)  # get minumum dist index
                        box = boxes[i]  # get box of face
                        if(min_dist < 1):
                             # get name corrosponding to minimum dist
                            name = name_list[min_dist_idx]              
                        else:
                            name = "Unknown"  # get name corrosponding to minimum dist
                            bbox = list(map(int,box.tolist()))
                            img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 2)
                            img = cv2.putText(img, name + '_{:.2f}'.format(min_dist), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_8)           
                            break
                # print("Name:", name)
                # print("Min dist:", min_dist)
                # print("Min dist idx:", min_dist_idx)
                # print("idx", idx)
                        box_peoples.append(box)
                        name_peoples.append(name)
                        min_dist_list.append(min_dist)
                #score = (torch.Tensor.cpu(min_dist.detach().numpy()*power))
                bbox = list(map(int,box.tolist()))
                img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 2)
                img = cv2.putText(img, name + '_{:.2f}'.format(min_dist), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_8)    
    data_peoples = [name_peoples, box_peoples,min_dist_list]
    return img,data_peoples

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download ="result.jpg">Download result</a>'
	return href

if st.button('Start Predict'):
    if uploaded_file is None : 
        st.write("Please upload an image")
    else:
        st.write("Start Predict")
        st.write("Reading image...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.write("Loaded data....")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        load_data = torch.load(os.path.join(dir_path,"data.pt")) 
        embedding_list = load_data[0] 
        name_list = load_data[1]
        imgpredict,data_peoples = predict_face(opencv_image)
        st.write("Predicting...")
        img = cv2.cvtColor(imgpredict, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (500, 660))
        st.image(img, caption=f"Image Predicted")
        result = Image.fromarray(img)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)
    