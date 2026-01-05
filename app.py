import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

model = YOLO("best.pt")

# Check what the actual class names are
print("Model class names:", model.names)
# This will show something like: {0: 'helmet', 1: 'no-helmet'} or {0: 'no-helmet', 1: 'helmet'}

def process_image(img):
    """Process single image for helmet detection"""
    if img is None:
        return None, "Please upload an image"
    
    try:
        # Convert PIL to BGR for OpenCV
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(img_array, verbose=False)
        
        helmet_count = 0
        no_helmet_count = 0
        
        # Draw boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # SWAPPED: class 0 is now NO HELMET, class 1 is HELMET
            if class_id == 1:  # Changed from 0 to 1 for Helmet
                color = (0, 255, 0)  # Green in BGR
                label = f"Helmet: {confidence:.2f}"
                helmet_count += 1
            else:  # class_id == 0 means No Helmet
                color = (0, 0, 255)  # Red in BGR
                label = f"No Helmet: {confidence:.2f}"
                no_helmet_count += 1
            
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_array, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Summary
        total = helmet_count + no_helmet_count
        if total > 0:
            info = f"👥 Total: {total}\n"
            info += f"🪖 Helmet: {helmet_count}\n"
            info += f"⚠️ No Helmet: {no_helmet_count}\n"
            if no_helmet_count > 0:
                info += f"\n🚨 Safety violation!"
            else:
                info += f"\n✅ All safe!"
        else:
            info = "No detections"
        
        return img_rgb, info
    
    except Exception as e:
        return None, f"Error: {e}"

def process_video(video_path):
    """Process video for helmet detection"""
    if video_path is None:
        return None, "Please upload a video"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        output_path = "output_helmet_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_helmet = 0
        total_no_helmet = 0
        violation_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            helmet_in_frame = 0
            no_helmet_in_frame = 0
            
            # Draw boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # SWAPPED: class 1 is Helmet, class 0 is No Helmet
                if class_id == 1:  # Changed from 0 to 1
                    color = (0, 255, 0)  # Green in BGR
                    label = f"Helmet: {confidence:.2f}"
                    helmet_in_frame += 1
                    total_helmet += 1
                else:  # class_id == 0
                    color = (0, 0, 255)  # Red in BGR
                    label = f"No Helmet: {confidence:.2f}"
                    no_helmet_in_frame += 1
                    total_no_helmet += 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Track violation frames
            if no_helmet_in_frame > 0:
                violation_frames += 1
            
            # Add statistics overlay
            status_color = (0, 0, 255) if no_helmet_in_frame > 0 else (0, 255, 0)
            cv2.putText(frame, f"Frame: {frame_count+1}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Helmet: {helmet_in_frame} | No Helmet: {no_helmet_in_frame}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if not os.path.exists(output_path):
            return None, "Error: Video output file was not created"
        
        # Summary
        info = f"✅ Video processed!\n"
        info += f"📊 Statistics:\n"
        info += f"Total frames: {total_frames}\n"
        info += f"🪖 Helmet detections: {total_helmet}\n"
        info += f"⚠️ No helmet detections: {total_no_helmet}\n"
        info += f"🚨 Violation frames: {violation_frames}/{total_frames}\n"
        
        if total_no_helmet > 0:
            info += f"\n⚠️ Safety violations detected!"
        else:
            info += f"\n✅ Full compliance!"
        
        return output_path, info
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

# Create Gradio interface with tabs
with gr.Blocks(title="Helmet Detection") as demo:
    gr.Markdown("# 🪖 Helmet Safety Detection System")
    gr.Markdown("Detect helmets in images or videos for workplace safety compliance")
    
    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                image_button = gr.Button("🔍 Detect Helmets", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Detection Result")
                image_info = gr.Textbox(label="Safety Report", lines=6)
        
        image_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, image_info]
        )
        
        gr.Markdown("**Legend:** 🟢 Green Box = Helmet | 🔴 Red Box = No Helmet")
    
    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_button = gr.Button("🎬 Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Video")
                video_info = gr.Textbox(label="Processing Report", lines=8)
        
        video_button.click(
            fn=process_video,
            inputs=video_input,
            outputs=[video_output, video_info]
        )
        
        gr.Markdown("**Note:** Video processing may take a few minutes depending on length")

if __name__ == "__main__":
    demo.launch()