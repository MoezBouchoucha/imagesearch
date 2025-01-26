import faiss
import numpy as np
from typing import List, Dict, Generator, Union
from pathlib import Path
import json
import math 
from tqdm import tqdm 

class ImageSearchEngine:
    def __init__(
        self,
        encoder: callable = None,
        input_dim: int = None,
        quantizer: callable = None,
    ):
        """
        Initialize search engine.
        Args:
            encoder: Function to convert raw data to embeddings
            input_dim: Optional embedding dimension
            sample_input: Sample for dimension inference
        """
        self.index=None
        self.quantizer=quantizer
        self.encoder=encoder
        self.input_dim=input_dim
        self.trained=False
        self.metadata= []
        self.number_of_clusters=2

    def set_nprobe(self):
        pass
    
    def set_n_clusters(self,num_images):
        """
        Calculate optimal number of regions based on dataset size.
        Rule of thumb: sqrt(N) for small datasets, 4*sqrt(N) for larger ones
        """
        print(f"found {num_images}...")
        if num_images < 100:
            n_regions = max(1, int(math.sqrt(num_images)))  
        else:
            n_regions = min(int(4 * math.sqrt(num_images)), num_images // 2)
        print(f"setting n_regions to {n_regions}")
        return n_regions   

    def set_index_from_examples(self,training_data: List[any]):
        example=training_data[0]

        if self.input_dim==None:
            print(f"system input dimension is not set.\n setting is automatically...")
            self.auto_set_dimension(example)
            print(f"set input dimension to {self.input_dim}")

        self.number_of_clusters = self.set_n_clusters(len(training_data))
        quantizer = faiss.IndexFlatL2(self.input_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.input_dim, self.number_of_clusters, faiss.METRIC_L2)

    
    def auto_set_dimension(self, sample_input: any,force:bool = False) -> None:
        dim=self.encoder(sample_input).shape[1]
        print(f"found dimension of {dim}...")
        if (self.input_dim == None) or (force==True):
            print(f"setting dimension to {dim}")
            self.input_dim=dim

    def train_index(self, training_data: List[any]) -> None:
        """Train index on in-memory data"""
        features = np.stack([self.encoder(item)[0] for item in training_data])
        print(features.shape)
        self.index.train(features)
        self.trained=True

    def add_items(self, items: List[any], metadata: List[Dict]) -> None:
        """Add items with associated metadata dictionaries"""
        # features = np.stack([self.encoder(item) for item in items])
        for i in tqdm(items,desc="adding features to the index"):
            self.index.add(self.encoder(i).numpy())
        self.metadata.extend(metadata)

    def search(self, query: str, k: int = 5,nprobe:int=None, verbose:bool=True,**search_params) -> List[Dict]:
        """Search index returning list of {metadata, distance} dicts"""
        if nprobe==None :
             self.index.nprobe=5
        D, I = self.index.search(self.encoder(query), k)
        if verbose==1:
            for i, neighbors in enumerate(I):
                print(f"Query {i + 1}:")
                for idx,neighbor_idx in enumerate(neighbors):
                    print(f"  Nearest neighbor index: {neighbor_idx}, File path: {self.metadata[neighbor_idx]}, Distance: {D[0][idx]}")
        return D , I

    def save(self, base_path: Union[str, Path]) -> None:
        """Persist index to .index file and metadata to .json"""
        base_path = Path(base_path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(base_path.with_suffix(".index")))
        
        # Save metadata as JSON
        with open(base_path.with_suffix(".json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, base_path: Union[str, Path], encoder_ref: callable) -> None:
        """Load index and metadata from disk"""
        base_path = Path(base_path)
        
        # Load FAISS index
        index = faiss.read_index(str(base_path.with_suffix(".index")))
        
        # Load metadata
        with open(base_path.with_suffix(".json"), "r") as f:
            metadata = json.load(f)
        
        self.index = index
        self.metadata = metadata
        self.encoder = encoder_ref
        self.trained = True  
        
    def _encode(self, data: List[any]) -> np.ndarray:
        """Convert raw data to embeddings using encoder"""
        return self.encoder(data).numpy()


# Core Attributes (to be maintained in class):
# - encoder: Embedding generation function
# - index: FAISS index object
# - metadata: List[Dict] storing item metadata
# - dimension: int embedding size
# - _trained: bool indicating training status
