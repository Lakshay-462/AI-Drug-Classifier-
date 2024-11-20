# PharmaNet: AI-Driven Drug Mechanism Classifier


PharmaNet is an advanced neural network-based project designed to classify drug mechanisms from SMILES (Simplified Molecular Input Line Entry System) strings. This application demonstrates the use of artificial intelligence in pharmaceutical research, providing a robust framework for identifying mechanisms such as enzyme inhibitors, receptor agonists, and more.

## Features
#### 1) Neural Network Implementation: 
Custom-built in C++ with efficient activation functions, backpropagation, and training mechanisms.
#### 2) SMILES Feature Extraction: 
Converts SMILES strings into meaningful input features for the neural network.
#### 3) Drug Mechanism Prediction: 
Predicts mechanisms across five categories:
1. Enzyme inhibitor
2. Receptor agonist
3. Receptor antagonist
4. Ion channel modulator
5. DNA intercalator
#### 4) Dataset Integration: 
Includes synthetic datasets for training and testing.

## Prerequisites
Before running the project, ensure you have:
1) A C++ compiler supporting C++11 or higher (e.g., GCC, Clang, MSVC).
2) A terminal or IDE to compile and execute the program.

## Installation
1) Clone the repository:
```bash
git clone https://github.com/<your-username>/pharmanet.git
cd pharmanet
```
2) Place the dataset file `synthetic_drug_dataset.csv` in the project root directory.

## Dataset Format
The dataset should be a CSV file with the following structure:

```csv
SMILES,Mechanism  
C1CCCCC1,0  
CCOCC,1  
...
```
`SMILES`:The SMILES string representation of a molecule.

`Mechanism`: An integer representing the mechanism category (0-4).

## Usage
### Training and Testing
1) Compile the program:

```bash
g++ -o pharmanet main.cpp -std=c++11
```
2) Run the program:

```bash
./pharmanet
```
3) Modify the `test_smiles` string in the code to test with new input.

## Example
Input:

```mathematica
SMILES: CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC(=O)NC3=CC=C(C=C3)F
```
Output:

```yaml
Predicted mechanism: Receptor antagonist
```

## Project Structure
`main.cpp`: Contains the neural network implementation, feature extractor, and main logic.

`synthetic_drug_dataset.csv`: Sample dataset for training and testing.

`README.md`: Project documentation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## Acknowledgments:
1) SMILES string concepts: OpenSMILES
2) Inspiration for AI in drug discovery.
