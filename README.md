# ChalkStream-Capstone-Group62
ChalkStream is a collaborative learning platform that uses computer vision and machine learning to capture lecture-board content in real time. The application processes handwritten text, mathematical expressions, and diagrams through an integrated pipeline of image processing, segmentation, and OCR components.

# Technologies

Language: Python

Machine Learning: PyTorch, Hugging Face Transformers

Computer Vision: OpenCV

Text Recognition: Math OCR, TrOCR



# Key Features

Real-Time Processing: Integrates multiple processing stages to transform lecture-board content into digital representations.

Image Preprocessing and Noise Removal: Prepares input frames for downstream recognition.

Content Segmentation: Separates relevant content into regions for specialized processing.

Handwritten Text Recognition: Uses a fine-tuned TrOCR model to recognize handwritten text and convert it to text.

Mathematical Expression Recognition: Uses a fine-tuned Math OCR to recognize mathematical content and converts recognized expressions into LaTeX.

Diagram Vectorization: Converts supported diagram content into vector representations.

Pipeline Orchestration: Coordinates processing components to produce integrated outputs.

# My Contributions

Fine-tuned the microsoft/trocr-base-handwritten model for handwritten text recognition.

Contributed to the OCR pipeline and image preprocessing workflow.

Evaluated recognition performance using character error rate (CER), word error rate (WER), and accuracy metrics.

Collaborated with the team on integrating and evaluating the overall application.

# Evaluation

Handwriting recognition was evaluated using character error rate (CER), word error rate (WER), and accuracy metrics.


# Team

Contributor 1 — Cameron Bodkin-Dick

Contributor 2 — Sandy Alzabadani

Contributor 3 — Eric Cui

# Acknowledgements

Thanks to all project contributors and the creators of the open-source models, libraries, and datasets used in this project.

