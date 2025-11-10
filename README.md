# Efficient Verification and Falsification of ReLU Neural Barrier Certificates
This repository contains the official implementation of the paper **Efficient Verification and Falsification of ReLU Neural Barrier Certificates** accepted to AAAI 2026.

# Requirements
See the **requirements.txt** in Verification. **We note that this repository should be installed in Ubuntu 22.04.** For further use without reliance on **Drake**, we will update soon.

# Verification
```sh
cd Verification
```

Install packages via pip
   ```sh
   pip install -r requirements.txt
   ```
To run the verification, enter the corresponding system folder and edit the files **main.py and superp.py**. According to the neural network architecture, modify the variable name **model_name** in **main.py** to select the neural network model automatically based on the .pt filename, e.g., **arch3_1_32.pt**. 
In **superp.py**, adjust **N_H** and **D_H** to specify the number of layers (N_H) and the number of neurons per layer (D_H).

# Training
If you want to train the neural barrier certificate, see the folder **Training** and the instruction in https://github.com/zhaohj2017/HSCC20-Repeatability, where the **Training** part is adapted from.

# Contact

If you have any questions, suggestions, or issues, feel free to contact me at **xueyl@ios.ac.cn**.

# License
Distributed under the MIT License. See `LICENSE.txt` for more information.
