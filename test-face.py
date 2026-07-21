import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(320, 320))

# Generate random image
random_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Detect faces
faces = app.get(random_image)

# Print results
print(f"Number of faces detected: {len(faces)}")
for i, face in enumerate(faces):
    print(f"\nFace {i+1}:")
    print(f"Bounding box: {face.bbox}")
    print(f"Confidence: {face.det_score}")
    print(f"Embedding shape: {face.embedding.shape}")
