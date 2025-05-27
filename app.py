import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO,RTDETR
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Setup detectron2 logger
setup_logger()

CUSTOM_CLASSES = ['car', 'truck', 'pedestrian', 'bicyclist', 'light']

def load_detectron2_model(model_path, config_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", confidence_threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT: Set the number of output classes for your custom model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CUSTOM_CLASSES)

    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def detectron2_inference(predictor, image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Detectron2 expects BGR format for input images
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image)
    return outputs

def visualize_detectron2_results(image, outputs, cfg):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    temp_metadata_name = "temp_custom_inference_meta_" + str(np.random.randint(0, 1000000))
    temp_metadata = MetadataCatalog.get(temp_metadata_name)
    temp_metadata.set(thing_classes=CUSTOM_CLASSES)

    v = Visualizer(image_bgr[:, :, ::-1], metadata=temp_metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]
    return result_image

def extract_detectron2_detections(outputs):
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    detections = []

    class_names_for_labeling = CUSTOM_CLASSES

    for i in range(len(boxes)):
        
        class_name = class_names_for_labeling[classes[i]] if classes[i] < len(class_names_for_labeling) else f"unknown_class_{classes[i]}"
        detections.append({
            'class': class_name,
            'confidence': scores[i],
            'x1': boxes[i][0],
            'y1': boxes[i][1],
            'x2': boxes[i][2],
            'y2': boxes[i][3]
        })
    return detections

# --- Streamlit UI starts here ---
st.title("Object Detection Demo")

# Model selection
model_choice = st.selectbox(
    "Choose Model:",
    ("YOLOv12", "RT-DETR", "FastRCNN"),
    help="Select the detection model to use for inference"
)


predictor = None
cfg = None

if model_choice == "YOLOv12":
    model = YOLO(r"models/YOLOv12n.pt")
    st.info("🚀 Using YOLOv12 model for detection")
elif model_choice == "RT-DETR":
    model = RTDETR(r"models/RTDETR.pt")
    st.info("🔥 Using RT-DETR model for detection")
else: # FastRCNN
    try:
        model_path = r"models/FastRCNN.pth"
        predictor, cfg = load_detectron2_model(model_path)
        st.info("⚡ Using FastRCNN (Detectron2) model for detection")
    except Exception as e:
        st.error(f"Error loading FastRCNN model: {str(e)}")
        st.error("Please ensure Detectron2 is installed and model path is correct. "
                 "Also, verify that `FASTRCNN.pth` was trained with your 5 classes and "
                 "that `cfg.MODEL.ROI_HEADS.NUM_CLASSES` is correctly set in `load_detectron2_model`.")
        st.stop()

# Detection type selection
detection_type = st.radio("Choose detection type:", ("Image Detection", "Video Detection"))

if detection_type == "Image Detection":
    st.subheader("📸 Image Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
     
        st.image(Image.open(tmp_path), caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Detection"):
            with st.spinner("Running detection..."):
                if model_choice in ["YOLOv12", "RT-DETR"]:
                    results = model(tmp_path)  
                    result = results[0]        
                    
          
                    plotted_img = result.plot()      # returns numpy array with boxes
                    
                    # Show the image
                    st.image(plotted_img, caption="Detected Objects", use_container_width=True)
                    
               
                    st.subheader("📊 Detection Results")
                    
                
                    if result.boxes is not None and len(result.boxes) > 0:
              
                        df = result.to_df()
                        st.dataframe(df, use_container_width=True)
                        
                      
                        st.subheader("📋 Detection Summary")
                        for i, box in enumerate(result.boxes):
                            conf = box.conf.item() if box.conf is not None else 0
                            cls_id = int(box.cls.item()) if box.cls is not None else 0
                            
                 
                            if result.names and cls_id in result.names:
                                class_name = result.names[cls_id]
                            else:
                                class_name = f"Unknown Class ID {cls_id}"
                                st.warning(f"Warning: Model predicted class ID {cls_id}, but no name found in model's class list.")
                          
                            
                            st.write(f"🎯 **Detection {i+1}:** {class_name} (Confidence: {conf:.2f})")
                    else:
                        st.warning("No objects detected in the image.")
                else:
                    image = cv2.imread(tmp_path)
                    outputs = detectron2_inference(predictor, image)

                    result_image = visualize_detectron2_results(image, outputs, cfg)
                    st.image(result_image, caption="Detected Objects", use_container_width=True)

                    detections = extract_detectron2_detections(outputs)

                    st.subheader("📊 Detection Results")
                    if detections:
                        df = pd.DataFrame(detections)
                        st.dataframe(df, use_container_width=True)

                        st.subheader("📋 Detection Summary")
                        for i, detection in enumerate(detections):
                            st.write(f"🎯 **Detection {i+1}:** {detection['class']} (Confidence: {detection['confidence']:.2f})")
                    else:
                        st.warning("No objects detected in the image.")
            
           
            os.remove(tmp_path) 

elif detection_type == "Video Detection":
    st.subheader("🎬 Video Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            tmp_video_path = tmp_video.name

        cap = cv2.VideoCapture(tmp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("Frames", f"{frame_count}")

        cap.release()

        st.subheader("⚙️ Processing Options")
        col1, col2 = st.columns(2)
        with col1:
            sample_rate = st.slider("Sample every N frames:", 1, 60, 15)
        with col2:
            conf_threshold = st.slider("Confidence Threshold:", 0.1, 1.0, 0.5, 0.1)

        if st.button("🎥 Run Video Detection"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            cap = cv2.VideoCapture(tmp_video_path)
            detection_data = []
            frame_idx = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    progress = frame_idx / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"🔍 Analyzing frame {frame_idx}/{frame_count}")

                    timestamp = frame_idx / fps

                    if model_choice in ["YOLOv12", "RT-DETR"]:
                        results = model(frame, conf=conf_threshold)
                        result = results[0]

                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                conf = box.conf.item() if box.conf is not None else 0
                                cls_id = int(box.cls.item()) if box.cls is not None else 0
                                
                                # --- FIX FOR KEYERROR: 7 ---
                                if result.names and cls_id in result.names:
                                    class_name = result.names[cls_id]
                                else:
                                    class_name = f"Unknown Class ID {cls_id}"
                                    st.warning(f"Warning: Model predicted class ID {cls_id}, but no name found in model's class list.")
                                # --- END FIX ---

                                detection_data.append({
                                    'timestamp': timestamp,
                                    'class': class_name,
                                    'confidence': conf,
                                    'x1': box.xyxy[0][0].item(),
                                    'y1': box.xyxy[0][1].item(),
                                    'x2': box.xyxy[0][2].item(),
                                    'y2': box.xyxy[0][3].item()
                                })
                    else: # FastRCNN (Detectron2)
                        outputs = detectron2_inference(predictor, frame)
                        instances = outputs["instances"].to("cpu")
                        scores = instances.scores.numpy()
                        classes = instances.pred_classes.numpy()
                        boxes = instances.pred_boxes.tensor.numpy()

                        # Use the global CUSTOM_CLASSES list directly for labeling
                        class_names_for_video = CUSTOM_CLASSES

                        for i, conf in enumerate(scores):
                            if conf >= conf_threshold:
                                class_name = class_names_for_video[classes[i]] if classes[i] < len(class_names_for_video) else f"unknown_class_{classes[i]}"
                                detection_data.append({
                                    'timestamp': timestamp,
                                    'class': class_name,
                                    'confidence': conf,
                                    'x1': boxes[i][0],
                                    'y1': boxes[i][1],
                                    'x2': boxes[i][2],
                                    'y2': boxes[i][3]
                                })

                    processed_frames += 1

                frame_idx += 1

            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video processing completed")

            if detection_data:
                df = pd.DataFrame(detection_data)
                st.subheader("📊 Detection Results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv, "detection_results.csv", "text/csv")
            else:
                st.warning("No detections found in the video.")

        os.remove(tmp_video_path)