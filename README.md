# SafeCLIP

SafeCLIP provides a lightweight zero-shot defense mechanism against toxic images by leveraging the inherent multimodal alignment in Large Vision-Language Models (LVLMs). This repository contains our implementation, built to efficiently detect harmful visual inputs with minimal system overhead—utilizing only publicly available models from Hugging Face.

---

## Overview

Large Vision-Language Models (LVLMs) have achieved remarkable advances in multimodal understanding through extensive pre-training and fine-tuning on large-scale datasets. However, their vulnerability to harmful visual inputs has raised safety concerns. SafeCLIP addresses this critical vulnerability without resorting to costly pre-filtering or fine-tuning methods.

**Key Features:**

- **Zero-Shot Toxic Image Detection:** Leverages the discarded CLS token from CLIP by projecting it into the text space and matching with toxic descriptors.
- **Efficiency & Low Overhead:** Achieves a 66.9% defense success rate with a low 3.2% false positive rate and only 7.2% overhead.
- **Dynamic Safety Corrections:** Enables dynamic adjustments during both inference and fine-tuning, without necessitating any architectural modifications.
- **Publicly Available Models:** Entirely built and tested using publicly available models from Hugging Face.

---

## Repository Structure

- **`models/`**  
  Contains our implementation of the SafeCLIP model. This is the core component that performs the zero-shot defense against toxic images.

- **`templates/`**  
  Provides customizable safety templates. These templates allow users to tailor the set of toxic descriptors and related settings to better suit their specific use case or threat model.

- **`notebooks/`**  
  Includes two Jupyter Notebook files that guide you through:
  - Running SafeCLIP.
  - Testing its performance on sample inputs.
  
  These notebooks provide step-by-step instructions and code examples to help you quickly integrate and validate the model in your workflow.

---

## Getting Started

### Prerequisites

Ensure you have installed:
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)
- Internet access for downloading models from Hugging Face

### Quick Start

1. **Clone the repository:**
   ```bash
   cd safeclip
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   - Follow the instructions to experiment with SafeCLIP and observe its performance in detecting toxic images.

---

## Usage

SafeCLIP leverages the seamless integration of LVLMs with publicly available models from Hugging Face. It requires no modifications to model architecture or heavy fine-tuning, making it a practical solution for dynamic and lightweight image toxicity filtering in various applications.

Feel free to adjust the safety templates provided in the `templates/` folder to suit your specific requirements.

---

## Contributing

We welcome contributions! If you have suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

---

---

## Acknowledgments

Special thanks to the community and developers behind Hugging Face and other open-source initiatives that made this work possible.
