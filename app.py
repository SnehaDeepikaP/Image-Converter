from flask import Flask, request, send_file, render_template_string, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import zipfile
import rarfile
import os
import io
import cv2
import numpy as np
import uuid
import shutil
import tempfile
from pathlib import Path
import pillow_heif
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import atexit

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)
CORS(app)

# Use temporary directory for better cleanup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR = tempfile.mkdtemp(prefix='ai_image_converter_')
UPLOAD_FOLDER = os.path.join(TEMP_DIR, 'uploads')
CONVERTED_FOLDER = os.path.join(TEMP_DIR, 'converted')

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# Global progress tracking
conversion_progress = {}
progress_lock = threading.Lock()

# Cleanup function for shutdown
def cleanup_temp_files():
    """Clean up temporary files on shutdown"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

# Register cleanup function
atexit.register(cleanup_temp_files)

@app.route('/')
def index():
    # Read the HTML file
    try:
        html_path = os.path.join(BASE_DIR, 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Please ensure index.html is in the same directory as app.py", 404

def enhance_image(image_pil, enhance_quality=True, adjust_brightness=True):
    """Apply AI-like enhancements to the image"""
    try:
        # Convert to RGB if needed
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        enhanced = image_pil
        
        if adjust_brightness:
            # Auto-adjust brightness based on image statistics
            img_array = np.array(enhanced)
            mean_brightness = np.mean(img_array)
            
            # Adjust brightness if too dark or too bright
            if mean_brightness < 100:  # Too dark
                brightness_factor = 1.3
            elif mean_brightness > 180:  # Too bright
                brightness_factor = 0.8
            else:
                brightness_factor = 1.1  # Slight enhancement
            
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness_factor)
        
        if enhance_quality:
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.2)
            
            # Enhance contrast slightly
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(1.1)
            
            # Enhance color saturation slightly
            color_enhancer = ImageEnhance.Color(enhanced)
            enhanced = color_enhancer.enhance(1.1)
            
            # Apply a subtle unsharp mask for better quality
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image_pil

def face_crop_centered(image_pil, target_width=None, target_height=None):
    """Improved face detection with proper centering of face in crop"""
    try:
        # Convert PIL image to OpenCV format
        img_array = np.array(image_pil)
        if len(img_array.shape) == 3:
            cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            cv_image = img_array
        
        # Convert to grayscale for face detection
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image
        
        print(f"Face detection on image size: {cv_image.shape}")
        
        # Try multiple cascade classifiers for better detection
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml'
        ]
        
        faces = []
        for cascade_file in cascade_files:
            try:
                cascade_path = cv2.data.haarcascades + cascade_file
                if os.path.exists(cascade_path):
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not face_cascade.empty():
                        # Try different parameters for better detection
                        for scale_factor in [1.05, 1.1, 1.15]:
                            for min_neighbors in [3, 4, 5]:
                                detected = face_cascade.detectMultiScale(
                                    gray, 
                                    scaleFactor=scale_factor, 
                                    minNeighbors=min_neighbors,
                                    minSize=(50, 50),  # Increased minimum size
                                    maxSize=(400, 400),  # Added maximum size
                                    flags=cv2.CASCADE_SCALE_IMAGE
                                )
                                if len(detected) > 0:
                                    faces = detected
                                    print(f"Found {len(faces)} faces with {cascade_file}")
                                    break
                            if len(faces) > 0:
                                break
                        if len(faces) > 0:
                            break
            except Exception as e:
                print(f"Error with cascade {cascade_file}: {e}")
                continue

        # If no face detected, return original image with custom dimensions if provided
        if len(faces) == 0:
            print("No faces detected, returning original image")
            if target_width and target_height:
                print(f"Resizing to custom dimensions: {target_width}x{target_height}")
                return image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return image_pil

        # Use the largest face for cropping
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        print(f"Selected face: x={x}, y={y}, w={w}, h={h}")
        
        # Calculate face center
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        print(f"Face center: ({face_center_x}, {face_center_y})")
        
        # Get image dimensions
        img_height, img_width = cv_image.shape[:2]
        
        # Determine crop dimensions
        if target_width and target_height:
            crop_width = target_width
            crop_height = target_height
        else:
            # Default crop size - make it proportional to face size
            crop_width = max(w * 3, 300)  # At least 3x face width, minimum 300px
            crop_height = max(h * 4, 400)  # At least 4x face height, minimum 400px
            
            # Ensure aspect ratio is reasonable (portrait)
            if crop_width > crop_height:
                crop_width = int(crop_height * 0.75)  # 3:4 aspect ratio
        
        print(f"Crop dimensions: {crop_width}x{crop_height}")
        
        # Calculate crop boundaries with face centered
        # For portraits, position face in upper third of image
        face_position_ratio = 0.4  # Face will be 40% from top (good for portraits)
        
        x1 = face_center_x - crop_width // 2
        y1 = face_center_y - int(crop_height * face_position_ratio)
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        # Adjust if crop goes outside image boundaries
        if x1 < 0:
            shift = -x1
            x1 = 0
            x2 = min(x2 + shift, img_width)
        elif x2 > img_width:
            shift = x2 - img_width
            x2 = img_width
            x1 = max(x1 - shift, 0)
            
        if y1 < 0:
            shift = -y1
            y1 = 0
            y2 = min(y2 + shift, img_height)
        elif y2 > img_height:
            shift = y2 - img_height
            y2 = img_height
            y1 = max(y1 - shift, 0)
        
        print(f"Final crop area: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Ensure we have a valid crop area
        if x2 > x1 and y2 > y1:
            # Crop the image
            cropped = cv_image[y1:y2, x1:x2]
            
            # Convert back to PIL Image
            if len(cropped.shape) == 3:
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            else:
                cropped_pil = Image.fromarray(cropped)
            
            print(f"Cropped image size: {cropped_pil.size}")
            
            # Apply custom dimensions if provided
            if target_width and target_height:
                print(f"Resizing cropped image to: {target_width}x{target_height}")
                cropped_pil = cropped_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            return cropped_pil
        else:
            print("Invalid crop area, returning original image")
            if target_width and target_height:
                return image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return image_pil
        
    except Exception as e:
        print(f"Error in face crop: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return original image with custom dimensions if provided
        if target_width and target_height:
            return image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image_pil

def process_single_image(args):
    """Process a single image - designed for parallel processing"""
    img_path, output_dir, convert_format, do_crop, enhance_quality, adjust_brightness, target_width, target_height, session_id = args
    
    try:
        print(f"Processing image: {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            raise Exception(f"File not found: {img_path}")
            
        # Load image (supports HEIC now)
        with Image.open(img_path) as img:
            print(f"Loaded image mode: {img.mode}, size: {img.size}")
            
            # Convert to RGB for processing
            if img.mode in ('RGBA', 'LA', 'P'):
                # Handle transparency properly
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Apply enhancements first (before cropping for better face detection)
            if enhance_quality or adjust_brightness:
                img = enhance_image(img, enhance_quality, adjust_brightness)
            
            # Apply face cropping and/or custom dimensions
            if do_crop:
                print("Applying face detection and cropping...")
                img = face_crop_centered(img, target_width, target_height)
            elif target_width and target_height:
                print(f"Resizing to custom dimensions: {target_width}x{target_height}")
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            print(f"Final image size: {img.size}")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Ensure proper extension mapping
            extension_map = {
                'jpg': 'jpg',
                'jpeg': 'jpg', 
                'png': 'png',
                'webp': 'webp'
            }
            
            file_extension = extension_map.get(convert_format.lower(), 'jpg')
            out_name = f"{base_name}.{file_extension}"
            output_path = os.path.join(output_dir, out_name)
            
            print(f"Saving to: {output_path}")
            
            # Prepare save arguments
            save_kwargs = {}
            save_format = convert_format.upper()
            
            # Handle format-specific settings
            if save_format in ['JPG', 'JPEG']:
                save_format = 'JPEG'  # PIL uses 'JPEG' not 'JPG'
                save_kwargs = {'quality': 95, 'optimize': True}
            elif save_format == 'PNG':
                save_kwargs = {'optimize': True}
            elif save_format == 'WEBP':
                save_kwargs = {'quality': 90, 'method': 6}
            
            # Save the image
            save_kwargs['dpi'] = (300, 300)
            img.save(output_path, save_format, **save_kwargs)
            print(f"Successfully saved: {output_path}")
        
        # Update progress
        with progress_lock:
            if session_id in conversion_progress:
                conversion_progress[session_id]['completed'] += 1
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error processing {os.path.basename(img_path)}: {str(e)}"
        print(error_msg)
        
        # Update progress with error
        with progress_lock:
            if session_id in conversion_progress:
                conversion_progress[session_id]['errors'].append(error_msg)
        
        return False, error_msg

def extract_archive(archive_path, extract_to):
    """Extract ZIP or RAR archive"""
    try:
        print(f"Extracting {archive_path} to {extract_to}")
        file_extension = os.path.splitext(archive_path.lower())[1]
        
        if file_extension == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_extension == '.rar':
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {file_extension}")
            return False
            
        print(f"Successfully extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload():
    session_id = str(uuid.uuid4())
    session_dir = None
    
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        zip_file = request.files['zipFile']
        if zip_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Get form parameters
        convert_format = request.form.get('format', 'jpg').lower()
        do_crop = request.form.get('crop_faces') == 'on'
        enhance_quality = request.form.get('enhance_quality') == 'on'
        adjust_brightness = request.form.get('adjust_brightness') == 'on'
        custom_dimensions = request.form.get('custom_dimensions') == 'on'
        
        target_width = 413
        target_height = 531
        if custom_dimensions:
            try:
                width_str = request.form.get('width', '').strip()
                height_str = request.form.get('height', '').strip()
                
                if width_str and height_str:
                    target_width = int(width_str)
                    target_height = int(height_str)
                    
                    if target_width <= 0 or target_height <= 0:
                        return jsonify({"error": "Width and height must be positive numbers"}), 400
                    if target_width > 4000 or target_height > 4000:
                        return jsonify({"error": "Maximum dimension is 4000 pixels"}), 400
            except (ValueError, TypeError) as e:
                return jsonify({"error": f"Invalid dimensions: {str(e)}"}), 400

        print(f"Starting conversion session: {session_id}")
        
        # Create session-specific directory
        session_dir = os.path.join(TEMP_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save uploaded archive
        file_extension = os.path.splitext(zip_file.filename)[1].lower()
        archive_path = os.path.join(session_dir, f"archive{file_extension}")
        zip_file.save(archive_path)

        # Extract archive
        extract_dir = os.path.join(session_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        if not extract_archive(archive_path, extract_dir):
            return jsonify({"error": "Error extracting archive. Supported formats: ZIP, RAR"}), 400

        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.heic', '.heif', '.gif'}
        image_files = []
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            return jsonify({"error": "No supported image files found in archive"}), 400
        
        print(f"Found {len(image_files)} images to process")
        
        # Initialize progress tracking
        with progress_lock:
            conversion_progress[session_id] = {
                'total': len(image_files),
                'completed': 0,
                'errors': []
            }
        
        # Create output directory
        output_dir = os.path.join(session_dir, 'converted')
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare arguments for parallel processing
        process_args = [
            (img_path, output_dir, convert_format, do_crop, enhance_quality, 
             adjust_brightness, target_width, target_height, session_id)
            for img_path in image_files
        ]
        
        # Process images in parallel with timeout protection
        max_workers = min(3, max(1, len(image_files) // 20))  # Reduced workers
        successful_conversions = 0
        
        print(f"Processing with {max_workers} workers")
        
        # Set a timeout for the entire processing
        start_time = time.time()
        timeout_seconds = 300  # 5 minutes maximum
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(process_single_image, args): args for args in process_args}
            
            for future in as_completed(future_to_args, timeout=timeout_seconds):
                try:
                    # Check for timeout
                    if time.time() - start_time > timeout_seconds:
                        print("Processing timeout reached")
                        break
                        
                    success, error = future.result(timeout=30)  # 30 seconds per image
                    if success:
                        successful_conversions += 1
                    elif error:
                        print(f"Processing error: {error}")
                except Exception as e:
                    print(f"Error processing future: {e}")
        
        print(f"Successfully converted {successful_conversions} out of {len(image_files)} images")
        
        # Get list of converted files
        converted_files = []
        if os.path.exists(output_dir):
            converted_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        
        if not converted_files:
            return jsonify({"error": "No images were successfully converted"}), 500
        
        # Create ZIP file in memory for immediate download
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for filename in converted_files:
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    zipf.write(file_path, arcname=filename)
        
        zip_buffer.seek(0)
        
        # Clean up progress tracking
        with progress_lock:
            if session_id in conversion_progress:
                del conversion_progress[session_id]
        
        # Generate download filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "face_cropped" if do_crop else "resized" if target_width else "converted"
        size_suffix = f"_{target_width}x{target_height}" if target_width and target_height else ""
        download_name = f'{prefix}_images_{len(converted_files)}files{size_suffix}_{timestamp}.zip'
        
        print(f"Sending download: {download_name}")
        
        # Return the file for download
        response = send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_name
        )
        
        # Schedule cleanup after response
        def cleanup_session():
            try:
                time.sleep(2)  # Wait a bit for download to start
                if session_dir and os.path.exists(session_dir):
                    shutil.rmtree(session_dir)
                    print(f"Cleaned up session directory: {session_dir}")
            except Exception as e:
                print(f"Error cleaning up session: {e}")
        
        threading.Thread(target=cleanup_session).start()
        
        return response
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if session_dir and os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
                print(f"Cleaned up session directory after error: {session_dir}")
            except Exception as cleanup_error:
                print(f"Error cleaning up session directory: {cleanup_error}")
        
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get conversion progress for a session"""
    with progress_lock:
        if session_id in conversion_progress:
            progress_data = conversion_progress[session_id].copy()
            progress_percent = (progress_data['completed'] / progress_data['total']) * 100 if progress_data['total'] > 0 else 0
            return jsonify({
                'progress': progress_percent,
                'completed': progress_data['completed'],
                'total': progress_data['total'],
                'errors': progress_data['errors'][-5:]  # Only return last 5 errors
            })
    return jsonify({'progress': 0, 'completed': 0, 'total': 0, 'errors': []})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'temp_dir': TEMP_DIR,
        'features': [
            'Improved face centering',
            'Faster processing',
            'Automatic cleanup',
            'Timeout protection'
        ]
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please ensure your archive is under 500MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error. Please try again."}), 500

if __name__ == '__main__':
    print("🚀 AI Image Converter Pro - Improved Version")
    print("✨ Features: Better face centering, faster processing, auto-cleanup")
    print(f"📂 Temp directory: {TEMP_DIR}")
    print("🌐 Server will be available at: http://localhost:5000")
    
    # Set maximum file size
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
