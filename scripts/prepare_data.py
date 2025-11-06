"""Data preparation script for face recognition dataset."""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from frs.core.alignment import FaceAligner
from frs.core.detector import FaceDetector
from frs.utils.config import config


def download_dataset(dataset_name: str, output_dir: Path):
    """Download face dataset.

    Args:
        dataset_name: Dataset name (vggface2, ms1m, lfw, etc.)
        output_dir: Output directory
    """
    logger.info(f"Downloading {dataset_name} dataset...")
    
    # Placeholder for dataset download logic
    # In practice, you would use APIs or scripts to download:
    # - WIDER FACE: http://shuoyang1213.me/WIDERFACE/
    # - VGGFace2: https://github.com/ox-vgg/vgg_face2
    # - MS-Celeb-1M (cleaned): https://github.com/EB-Dodo/C-MS-Celeb
    # - LFW: http://vis-www.cs.umass.edu/lfw/
    
    logger.warning(
        "Dataset download not implemented. "
        "Please manually download datasets to data/raw/ directory.\n"
        "Recommended structure:\n"
        "  data/raw/\n"
        "    identity_1/\n"
        "      image1.jpg\n"
        "      image2.jpg\n"
        "    identity_2/\n"
        "      image1.jpg\n"
    )


def collect_images(data_dir: Path) -> List[Tuple[Path, str]]:
    """Collect images and labels from directory.

    Args:
        data_dir: Directory containing identity subdirectories

    Returns:
        List of (image_path, identity_label) tuples
    """
    image_paths = []
    
    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        
        identity_name = identity_dir.name
        
        for img_path in identity_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append((img_path, identity_name))
    
    logger.info(f"Collected {len(image_paths)} images from {data_dir}")
    return image_paths


def process_image(
    image_path: Path,
    detector: FaceDetector,
    aligner: FaceAligner,
    output_dir: Path,
    identity_name: str,
    idx: int
) -> bool:
    """Process single image: detect, align, and save.

    Args:
        image_path: Input image path
        detector: Face detector instance
        aligner: Face aligner instance
        output_dir: Output directory
        identity_name: Identity label
        idx: Image index

    Returns:
        True if successful
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            return False
        
        # Detect faces
        faces = detector.detect(image)
        
        if len(faces) == 0:
            logger.debug(f"No face detected in {image_path}")
            return False
        
        # Use first face if multiple detected
        if len(faces) > 1:
            logger.debug(f"Multiple faces ({len(faces)}) detected in {image_path}, using first")
        
        face = faces[0]
        
        # Align face
        if 'landmarks' not in face or len(face['landmarks']) != 5:
            logger.debug(f"Insufficient landmarks for {image_path}")
            return False
        
        aligned_face = aligner.align(image, face['landmarks'])
        
        # Save aligned face
        identity_output = output_dir / identity_name
        identity_output.mkdir(parents=True, exist_ok=True)
        
        output_path = identity_output / f"{idx:04d}.jpg"
        cv2.imwrite(str(output_path), aligned_face)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return False


def create_train_val_split(
    processed_dir: Path,
    train_ratio: float = 0.8
) -> Tuple[List, List]:
    """Create train/val split maintaining identity distribution.

    Args:
        processed_dir: Directory with processed images
        train_ratio: Ratio of training data

    Returns:
        (train_list, val_list) containing (image_path, label) tuples
    """
    # Collect all images by identity
    identity_images = {}
    
    for identity_dir in sorted(processed_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        
        identity_name = identity_dir.name
        images = list(identity_dir.glob("*.jpg"))
        
        if len(images) > 0:
            identity_images[identity_name] = images
    
    logger.info(f"Found {len(identity_images)} identities")
    
    # Split each identity's images
    train_list = []
    val_list = []
    
    for identity_name, images in identity_images.items():
        if len(images) == 1:
            # If only one image, put in training set
            train_list.append((images[0], identity_name))
        else:
            # Split images for this identity
            train_imgs, val_imgs = train_test_split(
                images,
                train_size=train_ratio,
                random_state=42
            )
            
            train_list.extend([(img, identity_name) for img in train_imgs])
            val_list.extend([(img, identity_name) for img in val_imgs])
    
    logger.info(f"Train: {len(train_list)} images, Val: {len(val_list)} images")
    return train_list, val_list


def save_split_file(data_list: List[Tuple], output_file: Path):
    """Save train/val split to file.

    Args:
        data_list: List of (image_path, label) tuples
        output_file: Output JSON file
    """
    data = []
    for img_path, label in data_list:
        data.append({
            'image_path': str(img_path),
            'identity': label
        })
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved split to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare face recognition dataset")
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Raw dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed images'
    )
    parser.add_argument(
        '--download',
        type=str,
        default=None,
        help='Dataset to download (vggface2, ms1m, lfw)'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Training data ratio'
    )
    parser.add_argument(
        '--skip_detection',
        action='store_true',
        help='Skip face detection and alignment (if already processed)'
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    # Download dataset if requested
    if args.download:
        download_dataset(args.download, raw_dir)
    
    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_detection:
        # Initialize detector and aligner
        logger.info("Initializing detector and aligner...")
        detector = FaceDetector()
        aligner = FaceAligner()
        
        # Collect images
        image_list = collect_images(raw_dir)
        
        if len(image_list) == 0:
            logger.error(f"No images found in {raw_dir}")
            return
        
        # Process images
        logger.info("Processing images (detection and alignment)...")
        success_count = 0
        
        for idx, (img_path, identity_name) in enumerate(tqdm(image_list)):
            success = process_image(
                img_path,
                detector,
                aligner,
                output_dir,
                identity_name,
                idx
            )
            if success:
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(image_list)} images")
    
    # Create train/val split
    logger.info("Creating train/val split...")
    train_list, val_list = create_train_val_split(output_dir, args.train_split)
    
    # Save split files
    save_split_file(train_list, output_dir.parent / 'train.json')
    save_split_file(val_list, output_dir.parent / 'val.json')
    
    logger.info("Data preparation complete!")
    logger.info(f"  Processed images: {output_dir}")
    logger.info(f"  Train split: {output_dir.parent / 'train.json'}")
    logger.info(f"  Val split: {output_dir.parent / 'val.json'}")


if __name__ == '__main__':
    main()
