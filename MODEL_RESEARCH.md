# SIH26227 model research map

PS26227 asks for semantic retrieval plus multi-temporal change analysis,
false-alarm suppression, discovery/clustering, analyst review, provenance,
and offline/on-premise operation.

Recommended prototype stack:

A. Temporal change
   - ChangeFormerV6 (LEVir-CD)
   - BIT-CD
   - SNUNet-CD
   - classical co-registered pixel/edge difference as an explainable baseline

B. Structural / building evidence
   - U-Net
   - DeepLabV3+
   - SegFormer
   - SAM/SAMGeo for analyst-driven segmentation

C. Object evidence
   - YOLO/xView models for overhead objects
   - keep object detection separate from temporal change detection

D. Semantic retrieval
   - RemoteCLIP
   - FAISS for local vector indexing

E. False-alarm suppression
   - registration quality
   - cloud/haze/shadow masks
   - same-season comparison
   - sensor/viewing metadata
   - confidence thresholding
   - analyst review

F. Discovery
   - embeddings + FAISS
   - clustering of visually similar tiles/sites

For the first Streamlit demo, A + B + E + analyst review gives the clearest
judge-facing workflow. Retrieval and incremental archive indexing can be added
as the next module.
