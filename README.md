# Image Search Engine (Alpha Preview)

> **Development Notice**: This is an experimental preview version. Core functionality is implemented but needs optimization.  
> _Contains AI-assisted code implementations for critical components_

```text
Project Tree (v0.1-alpha)
├── data/                  # Image dataset directory
├── test/                  # Test images & validation
│
├── imagesearch.py         # Core FAISS engine (ImageSearchEngine)
├── test_inf.py            # Streamlit interface & inference
│
├── image_db.index         # Example FAISS index
├── image_db.json          # Sample metadata
│
├── requirements.txt       # Dependency spec
└── README.md              # This document
```

# Image Search Engine (Alpha Preview)

> **Experimental Version**: Early development stage - core features functional but unoptimized

## Quick Start 🚀

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Test with Sample Data

```bash
# Prepare sample images
mkdir -p data/images
cp test/*.jpg data/images/

# Launch web interface
streamlit run test_inf.py
```

## Key Components 🧩

### `imagesearch.py`

- **FAISS-powered search backend**
- Core class: `ImageSearchEngine`
  - Index creation/loading
  - Metadata management

### `test_inf.py`

- **Streamlit web interface**
  - Vision Transformer (ViT) embeddings
  - Drag-and-drop image upload
  - Similarity results grid
  - Temporary file handling
  - Interactive previews

### Pre-built Index (`image_db.*`)

- **Demo database**
  - `image_db.index`: FAISS vector index
  - `image_db.json`: Image paths & metadata

## Development Roadmap 🗺️

| Status | Feature                   |
| ------ | ------------------------- |
| ✓      | Core FAISS Integration    |
| ✓      | Basic Streamlit Interface |
| ◌      | Performance Optimization  |
| ◌      | Batch Indexing Tools      |
| ◌      | Cloud Deployment Setup    |

## Configuration Guide ⚙️

| File               | Purpose                        |
| ------------------ | ------------------------------ |
| `image_db.index`   | Binary FAISS vector index      |
| `image_db.json`    | Image paths & metadata storage |
| `requirements.txt` | Python dependencies            |

## Contribution Notes 📝

### Current Limitations

- Single-machine indexing
- Basic error handling
- Manual dataset preparation

> **Alpha Notice**: This preview version demonstrates core functionality but requires optimization for production use.

```

```
