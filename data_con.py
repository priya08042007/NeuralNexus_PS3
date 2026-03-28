import os
import cv2

classes = ['fiber', 'fragment', 'film', 'pellet']

src = "DatasetB/val"
dst_img = "final_dataset/val/images"
dst_lbl = "final_dataset/val/labels"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

for cls_id, cls_name in enumerate(classes):
    cls_path = os.path.join(src, cls_name)
    
    if not os.path.exists(cls_path):
        continue
    
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        
        # Copy image
        new_name = f"B_{img_name}"
        cv2.imwrite(os.path.join(dst_img, new_name), cv2.imread(img_path))
        
        # Create label (FULL IMAGE box)
        label_path = os.path.join(dst_lbl, new_name.replace('.jpg', '.txt'))
        
        with open(label_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

print("Dataset consolidation completed.")