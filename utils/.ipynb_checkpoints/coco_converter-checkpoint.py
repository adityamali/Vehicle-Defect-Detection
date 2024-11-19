# utils/coco_converter.py
import json
import os
from pathlib import Path
import shutil
import yaml

def convert_coco_to_yolo(coco_json_path, output_dir, image_dir=None):
    try:
        # Create output directory structure
        output_dir = Path(output_dir)
        labels_dir = output_dir / 'labels'
        os.makedirs(labels_dir, exist_ok=True)
        
        if image_dir:
            images_out_dir = output_dir / 'images'
            os.makedirs(images_out_dir, exist_ok=True)
        
        # Load and validate COCO JSON
        with open(coco_json_path, 'r') as f:
            try:
                coco_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON file")
        
        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        if not all(field in coco_data for field in required_fields):
            raise ValueError(f"Missing required fields in COCO JSON: {required_fields}")
            
        # Create category mappings
        categories = coco_data['categories']
        cat_id_to_idx = {}
        names_dict = {}
        
        # Create both mappings in one loop
        for idx, cat in enumerate(categories):
            cat_id_to_idx[cat['id']] = idx
            names_dict[idx] = cat['name']
        
        # Group annotations by image ID
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Process each image
        converted_count = 0
        for img in coco_data['images']:
            img_id = img['id']
            img_width = img['width']
            img_height = img['height']
            
            # Create YOLO format annotations
            yolo_anns = []
            if img_id in img_to_anns:
                for ann in img_to_anns[img_id]:
                    # Get category index
                    cat_idx = cat_id_to_idx[ann['category_id']]
                    
                    # Get bbox coordinates
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    w = w / img_width
                    h = h / img_height
                    
                    # Ensure coordinates are within bounds
                    if not all(0 <= coord <= 1 for coord in [x_center, y_center, w, h]):
                        print(f"Warning: Invalid coordinates in image {img_id}")
                        continue
                    
                    yolo_anns.append(f"{cat_idx} {x_center} {y_center} {w} {h}")
            
            # Save YOLO format annotations
            base_filename = Path(img['file_name']).stem
            label_path = labels_dir / f"{base_filename}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_anns))
            
            # Copy image if image_dir is provided
            if image_dir:
                src_img_path = Path(image_dir) / img['file_name']
                if src_img_path.exists():
                    shutil.copy2(src_img_path, images_out_dir / img['file_name'])
                else:
                    print(f"Warning: Image file not found: {src_img_path}")
            
            converted_count += 1
        
        # Create dataset.yaml using the names_dict
        yaml_content = {
            'path': str(output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'names': names_dict
        }
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"Successfully converted {converted_count} images")
        print(f"Total categories: {len(names_dict)}")
        print(f"Category names: {names_dict}")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise