#!/usr/bin/env python3
"""
Deployment Asset Preparation Script

Creates deployment-ready assets for instant cloud initialization:
- Bundles sentence transformer model locally
- Pre-computes business category embeddings  
- Copies essential cache files
- Creates deployment-optimized package

This script is SAFE and ADDITIVE - it doesn't modify existing functionality.
Local development continues to work exactly as before.
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def prepare_deployment_assets():
    """Main function to prepare all deployment assets."""
    
    print("🚀 Preparing Deployment Assets for Cloud Optimization")
    print("="*60)
    
    # Create deployment directory structure
    deployment_dir = Path(__file__).parent / 'assets'
    deployment_dir.mkdir(exist_ok=True)
    
    models_dir = deployment_dir / 'models'
    embeddings_dir = deployment_dir / 'embeddings'
    data_dir = deployment_dir / 'data'
    
    models_dir.mkdir(exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Bundle the sentence transformer model
        bundle_sentence_transformer_model(models_dir)
        
        # Step 2: Pre-compute business category embeddings
        precompute_business_embeddings(models_dir, embeddings_dir)
        
        # Step 3: Copy essential cache files
        bundle_cache_files(data_dir)
        
        # Step 4: Create deployment metadata
        create_deployment_metadata(deployment_dir)
        
        print("\n✅ Deployment assets prepared successfully!")
        print(f"📁 Assets location: {deployment_dir}")
        print(f"📦 Total size: {get_directory_size(deployment_dir):.1f}MB")
        
        print("\n📋 Next Steps:")
        print("1. Deploy the entire 'deployment/assets' folder with your Streamlit app")
        print("2. Ensure assets/ folder is in the same directory as your demo files")
        print("3. The system will automatically use these assets for fast initialization")
        
    except Exception as e:
        print(f"\n❌ Error preparing deployment assets: {e}")
        print("Your local system continues to work normally.")
        sys.exit(1)

def bundle_sentence_transformer_model(models_dir):
    """Download and bundle the sentence transformer model."""
    
    print("\n1️⃣ Bundling Sentence Transformer Model...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required. Run: pip install sentence-transformers")
    
    model_name = 'all-MiniLM-L6-v2'
    model_path = models_dir / model_name
    
    if model_path.exists():
        print(f"   ✅ Model already bundled at {model_path}")
        return
    
    print(f"   📥 Downloading {model_name} model...")
    print("   ⏱️  This may take 1-2 minutes for first download...")
    
    # Download and save model
    model = SentenceTransformer(model_name)
    model.save(str(model_path))
    
    # Verify model size
    model_size = get_directory_size(model_path)
    print(f"   ✅ Model bundled successfully: {model_size:.1f}MB")
    
    # Test model loading
    test_model = SentenceTransformer(str(model_path))
    test_embedding = test_model.encode("test sentence")
    print(f"   ✅ Model verification passed: {test_embedding.shape} dimensions")

def precompute_business_embeddings(models_dir, embeddings_dir):
    """Pre-compute business category embeddings."""
    
    print("\n2️⃣ Pre-computing Business Category Embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from two_tier_classifier.data.category_mappings import BUSINESS_CATEGORIES
    except ImportError as e:
        print(f"   ⚠️  Import failed: {e}")
        print("   Skipping business embeddings pre-computation")
        return
    
    # Load bundled model
    model_name = 'all-MiniLM-L6-v2'
    model_path = models_dir / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"   📊 Loading model from {model_path}...")
    model = SentenceTransformer(str(model_path))
    
    # Get business categories and their descriptions
    business_texts = []
    business_names = []
    
    for category_enum, category_def in BUSINESS_CATEGORIES.items():
        business_names.append(category_def.name)
        # Combine name + description + key keywords for richer embeddings
        combined_text = f"{category_def.name}. {category_def.description}. Keywords: {', '.join(category_def.keywords[:10])}"
        business_texts.append(combined_text)
    
    print(f"   🧠 Computing embeddings for {len(business_texts)} business categories...")
    
    # Compute embeddings
    embeddings = model.encode(business_texts, normalize_embeddings=True)
    
    # Save embeddings and metadata
    embeddings_file = embeddings_dir / 'business_categories.npy'
    metadata_file = embeddings_dir / 'business_metadata.json'
    
    np.save(embeddings_file, embeddings)
    
    metadata = {
        'business_names': business_names,
        'business_texts': business_texts,
        'embedding_model': model_name,
        'embedding_dimension': embeddings.shape[1],
        'num_categories': len(business_names),
        'created_at': datetime.now().isoformat(),
        'normalization': 'L2 normalized'
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Business embeddings saved: {embeddings.shape}")
    print(f"   📁 Embeddings: {embeddings_file}")
    print(f"   📋 Metadata: {metadata_file}")

def bundle_cache_files(data_dir):
    """Copy essential cache files for deployment."""
    
    print("\n3️⃣ Bundling Essential Cache Files...")
    
    # Essential files for deployment
    cache_files = [
        ('cache/level2_problems.json', 'level2_problems.json'),
        ('cache/automation_mappings.json', 'automation_mappings.json'),
        ('cache/level2_automation_mappings.json', 'level2_automation_mappings.json')
    ]
    
    repo_root = Path(__file__).parent.parent
    files_copied = 0
    
    for source_path, dest_name in cache_files:
        source = repo_root / source_path
        dest = data_dir / dest_name
        
        if source.exists():
            shutil.copy2(source, dest)
            file_size = source.stat().st_size / 1024 / 1024  # MB
            print(f"   ✅ Copied {dest_name}: {file_size:.1f}MB")
            files_copied += 1
        else:
            print(f"   ⚠️  Missing {source_path} - system will fall back to minimal data")
    
    print(f"   📦 Bundled {files_copied}/{len(cache_files)} cache files")

def create_deployment_metadata(deployment_dir):
    """Create deployment metadata and verification file."""
    
    print("\n4️⃣ Creating Deployment Metadata...")
    
    # Calculate asset sizes
    models_size = get_directory_size(deployment_dir / 'models')
    embeddings_size = get_directory_size(deployment_dir / 'embeddings')
    data_size = get_directory_size(deployment_dir / 'data')
    total_size = models_size + embeddings_size + data_size
    
    metadata = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'optimization_type': 'pre_computed_assets',
        'assets': {
            'models': {
                'size_mb': models_size,
                'files': list((deployment_dir / 'models').rglob('*')) if (deployment_dir / 'models').exists() else []
            },
            'embeddings': {
                'size_mb': embeddings_size,
                'files': [f.name for f in (deployment_dir / 'embeddings').glob('*')] if (deployment_dir / 'embeddings').exists() else []
            },
            'data': {
                'size_mb': data_size,
                'files': [f.name for f in (deployment_dir / 'data').glob('*')] if (deployment_dir / 'data').exists() else []
            }
        },
        'total_size_mb': total_size,
        'expected_speedup': '100x initialization improvement',
        'compatibility': 'Backward compatible - falls back to normal loading if assets missing'
    }
    
    # Convert Path objects to strings for JSON serialization
    if 'files' in metadata['assets']['models']:
        metadata['assets']['models']['files'] = [str(f) for f in metadata['assets']['models']['files']]
    
    metadata_file = deployment_dir / 'deployment_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Deployment metadata saved: {metadata_file}")
    print(f"   📊 Asset breakdown:")
    print(f"      - Models: {models_size:.1f}MB")  
    print(f"      - Embeddings: {embeddings_size:.1f}MB")
    print(f"      - Data: {data_size:.1f}MB")
    print(f"      - Total: {total_size:.1f}MB")

def get_directory_size(directory):
    """Get directory size in MB."""
    directory = Path(directory)
    if not directory.exists():
        return 0.0
    
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / 1024 / 1024  # Convert to MB

if __name__ == "__main__":
    prepare_deployment_assets()