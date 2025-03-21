import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'ppm'}

# Folder paths for visualization steps
VESSEL_SEG_FOLDER = 'static/H/vessel_segmentation'
AV_CLASS_FOLDER = 'static/H/av_classification'
FD_FOLDER = 'static/H/fd'

# CSV file path
RETINAL_DATA_CSV = 'static/H/retinal_dataset_FFinal.csv'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model paths - update these to your model paths
CUP_MODEL_PATH = "models/cup_model.pt"
DISC_MODEL_PATH = "models/disc_model.pt"

# Load models
cup_model = None
disc_model = None

def load_models():
    global cup_model, disc_model
    print("Loading models...")
    cup_model = YOLO(CUP_MODEL_PATH)
    disc_model = YOLO(DISC_MODEL_PATH)
    print("Models loaded successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_web_format(img_path):
    """Convert TIFF files to PNG for web display"""
    if img_path.lower().endswith(('.tif', '.tiff')):
        # Create a web-friendly version with PIL to avoid potential OpenCV TIFF issues
        try:
            img = Image.open(img_path)
            web_path = img_path.rsplit('.', 1)[0] + '.png'
            img.save(web_path)
            return web_path
        except Exception as e:
            print(f"Error converting TIFF to PNG: {str(e)}")
            return img_path
    return img_path

def calculate_cdr(img_path):
    """
    Calculate cup-to-disc ratio for a single image

    Args:
        img_path (str): Path to the input image

    Returns:
        tuple: (CDR value, cup mask image, disc mask image, overlay image)
    """
    try:
        # For TIFF images, use PIL to read the image
        if img_path.lower().endswith(('.tif', '.tiff')):
            pil_img = Image.open(img_path)
            img = np.array(pil_img)
            # Convert RGB if needed
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            # Regular read for non-TIFF images
            img = cv2.imread(img_path)
            
        if img is None:
            print(f"Could not read image: {img_path}")
            return -1, None, None, None

        H, W = img.shape[:2]
        original_img = img.copy()

        # Get segmentation results
        cup_results = cup_model(img)
        disc_results = disc_model(img)

        # Create mask images
        cup_mask_img = np.zeros((H, W), dtype=np.uint8)
        disc_mask_img = np.zeros((H, W), dtype=np.uint8)
        
        # Process cup mask
        cup_area = 0
        if len(cup_results) > 0 and hasattr(cup_results[0], 'masks') and cup_results[0].masks is not None:
            cup_mask = cup_results[0].masks.data[0].cpu().numpy()
            cup_mask = (cv2.resize(cup_mask, (W, H)) * 255).astype(np.uint8)
            cup_mask_img = cup_mask
            cup_area = np.sum(cup_mask > 0)

        # Process disc mask
        disc_area = 0
        if len(disc_results) > 0 and hasattr(disc_results[0], 'masks') and disc_results[0].masks is not None:
            disc_mask = disc_results[0].masks.data[0].cpu().numpy()
            disc_mask = (cv2.resize(disc_mask, (W, H)) * 255).astype(np.uint8)
            disc_mask_img = disc_mask
            disc_area = np.sum(disc_mask > 0)

        # Create visualization overlay
        overlay = original_img.copy()
        
        # Create colored masks for visualization
        cup_colored = np.zeros_like(original_img)
        cup_colored[cup_mask_img > 0] = [0, 0, 255]  # Red for cup
        
        disc_colored = np.zeros_like(original_img)
        disc_colored[disc_mask_img > 0] = [0, 255, 0]  # Green for disc
        
        # Overlay masks
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, disc_colored, alpha, 0)
        overlay = cv2.addWeighted(overlay, 1, cup_colored, alpha, 0)
        
        # Draw contours
        cup_contours, _ = cv2.findContours(cup_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        disc_contours, _ = cv2.findContours(disc_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cup_contours:
            cv2.drawContours(overlay, cup_contours, -1, (255, 255, 255), 2)
        if disc_contours:
            cv2.drawContours(overlay, disc_contours, -1, (255, 255, 255), 1)
            
        # Calculate ratio
        if disc_area == 0:
            print(f"No disc detected in image: {img_path}")
            return -1, cup_mask_img, disc_mask_img, overlay

        cdr = cup_area / disc_area
        return cdr, cup_mask_img, disc_mask_img, overlay

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1, None, None, None

def get_avr_and_fd(image_name):
    """
    Get AVR and Fractal Dimension values from CSV file
    
    Args:
        image_name (str): Image name to look up in CSV
        
    Returns:
        tuple: (AVR value, FD value)
    """
    try:
        # Remove file extension for comparison
        base_name = os.path.splitext(image_name)[0]
        print(base_name+'1')
        # Load CSV data
        df = pd.read_csv(RETINAL_DATA_CSV)
        
        # Look for the image in the CSV
        row = df[df['image_name'].str.contains(base_name, case=False, na=False)]
        
        if row.empty:
            print(f"Image {base_name} not found in CSV data")
            return None, None
            
        # Get AVR and FD values
        avr = float(row['AV'].values[0]) if not pd.isna(row['AV'].values[0]) else None
        fd = float(row['FractalDimensions'].values[0]) if not pd.isna(row['FractalDimensions'].values[0]) else None
        
        return avr, fd
        
    except Exception as e:
        print(f"Error getting AVR/FD for {image_name}: {str(e)}")
        return None, None

def get_process_images(image_name, folders):
    images = {}
    base_name = os.path.splitext(image_name)[0]
    for folder_key, folder_path in folders.items():
        if not os.path.exists(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[0] == base_name:
                full_path = os.path.join(folder_path, file)
                # Convert to web-friendly path
                web_path = os.path.relpath(full_path, 'static').replace("\\", "/")
                web_path = web_path.replace("\\", "/")  # Fix Windows paths
                # Remove any leading ../
                if web_path.startswith("../"):
                    web_path = web_path[3:]
                images[folder_key] = web_path
                break
    
    return images

def assess_risk(obm_present, avr, cdr, fd, nbm_present):
    """
    Assess CVD risk based on defined rules
    
    Args:
        obm_present (bool): If OBM present
        avr (float): AVR value
        cdr (float): CDR value
        fd (float): FD value
        nbm_present (bool): If NBM present
        
    Returns:
        tuple: (risk level, reason)
    """
    # Check if parameters are outside normal range
    avr_abnormal = avr is not None and (avr < 0.67 or avr > 0.75)
    cdr_abnormal = cdr is not None and cdr > 0.6
    fd_abnormal = fd is not None and fd < 1.3
    
    any_abnormal = avr_abnormal or cdr_abnormal or fd_abnormal
    
    # Apply risk rules
    if obm_present:
        return "High Risk", "Presence of high-risk ophthalmic biomarkers (OBM)"
    elif nbm_present and not any_abnormal:
        return "Low Risk", "Only neutral biomarkers with normal features"
    elif nbm_present and any_abnormal:
        reasons = []
        if avr_abnormal: reasons.append(f"abnormal AVR ({avr})")
        if cdr_abnormal: reasons.append(f"abnormal CDR ({cdr})")
        if fd_abnormal: reasons.append(f"abnormal FD ({fd})")
        return "High Risk", f"Neutral biomarkers with {' and '.join(reasons)}"
    elif any_abnormal:
        reasons = []
        if avr_abnormal: reasons.append(f"abnormal AVR ({avr})")
        if cdr_abnormal: reasons.append(f"abnormal CDR ({cdr})")
        if fd_abnormal: reasons.append(f"abnormal FD ({fd})")
        return "High Risk", f"No neutral biomarkers but with {' and '.join(reasons)}"
    else:
        return "Low Risk", "Normal features detected"
    
def apply_clahe(img_path):
    """Apply CLAHE to grayscale and return processed image path"""
    try:
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image")
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit
        clahe_img = clahe.apply(img)
        
        # Convert to RGB for web display (pseudo-color)
        colored_heatmap = cv2.applyColorMap(clahe_img, cv2.COLORMAP_BONE)
        
        # Save as PNG
        original_name = os.path.splitext(os.path.basename(img_path))[0]
        filename = f"clahe_{original_name}.png"
        save_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        cv2.imwrite(save_path, colored_heatmap)
        
        return filename
    except Exception as e:
        print(f"Error in CLAHE processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Get biomarker selections
    biomarkers = request.form.getlist('biomarkers[]')
    obm_conditions = [b.replace('obm_', '') for b in biomarkers if b.startswith('obm_')]
    nbm_conditions = [b.replace('nbm_', '') for b in biomarkers if b.startswith('nbm_')]
        
    obm_present = len(obm_conditions) > 0
    nbm_present = len(nbm_conditions) > 0
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make sure models are loaded
        if cup_model is None or disc_model is None:
            load_models()
        
        # Process the image for CDR
        cdr, cup_mask, disc_mask, overlay = calculate_cdr(file_path)
        
        if cdr == -1:
            flash('Failed to process image. Please try another image.')
            return redirect(url_for('index'))
        
        # Get AVR and FD from CSV
        avr, fd = get_avr_and_fd(filename)
        
        # Get process images for visualization
        process_folders = {
            'vessel_seg': VESSEL_SEG_FOLDER,
            'av_class': AV_CLASS_FOLDER,
            'fd': FD_FOLDER
        }
        
        process_images = get_process_images(filename, process_folders)
        
        # Assess risk
        risk_level, risk_reason = assess_risk(obm_present, avr, cdr, fd, nbm_present)
        
        # Create web-friendly versions of all images if needed
        web_friendly_name = filename
        # If it's a TIFF, convert the original too
        if filename.lower().endswith(('.tif', '.tiff')):
            img = Image.open(file_path)
            web_friendly_name = filename.rsplit('.', 1)[0] + '.png'
            web_path = os.path.join(app.config['UPLOAD_FOLDER'], web_friendly_name)
            img.save(web_path)
        
        # Save result images for CDR - always as PNG
        cup_filename = f'cup_{web_friendly_name.rsplit(".", 1)[0]}.png'
        disc_filename = f'disc_{web_friendly_name.rsplit(".", 1)[0]}.png'
        overlay_filename = f'overlay_{web_friendly_name.rsplit(".", 1)[0]}.png'
        
        cup_vis_path = os.path.join(app.config['RESULTS_FOLDER'], cup_filename)
        disc_vis_path = os.path.join(app.config['RESULTS_FOLDER'], disc_filename)
        overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
        
        # Create colored visualizations for individual masks
        cup_vis = np.zeros((cup_mask.shape[0], cup_mask.shape[1], 3), dtype=np.uint8)
        cup_vis[cup_mask > 0] = [0, 0, 255]  # Red for cup
        
        disc_vis = np.zeros((disc_mask.shape[0], disc_mask.shape[1], 3), dtype=np.uint8)
        disc_vis[disc_mask > 0] = [0, 255, 0]  # Green for disc
        
        # Save CDR result images
        cv2.imwrite(cup_vis_path, cup_vis)
        cv2.imwrite(disc_vis_path, disc_vis)
        cv2.imwrite(overlay_path, overlay)
        
        # Generate paths for template
        web_friendly_original = web_friendly_name if filename.lower().endswith(('.tif', '.tiff')) else filename
        
        # Prepare process image paths for template
        template_process_images = {}
        for key, path in process_images.items():
            if path:
                # Convert to relative path for template
                template_process_images[key] = path

        # Apply CLAHE and save result
        clahe_filename = apply_clahe(file_path)
        print(file_path)
        print(clahe_filename)
        if not clahe_filename:
            flash('Error in image preprocessing')
            return redirect(url_for('index'))
        
        # Return results
        return render_template('cvd_result.html', 
            original_image=f"uploads/{web_friendly_original}",
            cup_image=f"results/{cup_filename}",
            disc_image=f"results/{disc_filename}",
            overlay_image=f"results/{overlay_filename}",
            cdr=round(cdr, 3),
            avr=round(avr, 3) if avr is not None else "N/A",
            fd=round(fd, 3) if fd is not None else "N/A",
            clahe_image=f"results/{clahe_filename}",
            risk_level=risk_level,
            risk_reason=risk_reason,
            process_images=template_process_images,
            obm_present=obm_present,
            nbm_present=nbm_present)
    
    flash('File type not allowed')
    return redirect(request.url)

if __name__ == '__main__':
    # Load models at startup
    load_models()
    app.run(debug=True)