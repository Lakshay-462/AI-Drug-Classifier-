#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;

// Neural Network implementation in pure C++
class NeuralNetwork {
private:
    struct Layer {
        vector<vector<double> >  weights;
        vector<double> biases;
        vector<double> neurons;
        vector<double> deltas;
    };

    vector<Layer> layers;
    double learning_rate;

    double activate(double x) {
        return max(0.0, x);
    }

    double activate_derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    void initializeWeights(int input_size, int hidden_size, int output_size) {
        random_device rd;
        mt19937 gen(rd());

        // Initialize hidden layer
        Layer hidden;
        double scale = sqrt(2.0 / input_size);
        normal_distribution<double> d(0, scale);

        hidden.weights.resize(hidden_size, vector<double>(input_size));
        hidden.biases.resize(hidden_size);
        hidden.neurons.resize(hidden_size);
        hidden.deltas.resize(hidden_size);

        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                hidden.weights[i][j] = d(gen);
            }
            hidden.biases[i] = 0;
        }
        layers.push_back(hidden);

        // Initialize output layer
        Layer output;
        scale = sqrt(2.0 / hidden_size);
        normal_distribution<double> d2(0, scale);

        output.weights.resize(output_size, vector<double>(hidden_size));
        output.biases.resize(output_size);
        output.neurons.resize(output_size);
        output.deltas.resize(output_size);

        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                output.weights[i][j] = d2(gen);
            }
            output.biases[i] = 0;
        }
        layers.push_back(output);
    }

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr = 0.01)
        : learning_rate(lr) {
        initializeWeights(input_size, hidden_size, output_size);
    }

    vector<double> forward(const vector<double>& input) {
        // Input to hidden layer
        for (int i = 0; i < layers[0].neurons.size(); i++) {
            double sum = layers[0].biases[i];
            for (int j = 0; j < input.size(); j++) {
                sum += input[j] * layers[0].weights[i][j];
            }
            layers[0].neurons[i] = activate(sum);
        }

        // Hidden to output layer
        for (int i = 0; i < layers[1].neurons.size(); i++) {
            double sum = layers[1].biases[i];
            for (int j = 0; j < layers[0].neurons.size(); j++) {
                sum += layers[0].neurons[j] * layers[1].weights[i][j];
            }
            layers[1].neurons[i] = activate(sum);
        }

        return layers[1].neurons;
    }

    void backward(const vector<double>& input, const vector<double>& target) {
        // Calculate output layer deltas
        for (int i = 0; i < layers[1].neurons.size(); i++) {
            double output = layers[1].neurons[i];
            layers[1].deltas[i] = (target[i] - output) * activate_derivative(output);
        }

        // Calculate hidden layer deltas
        for (int i = 0; i < layers[0].neurons.size(); i++) {
            double error = 0.0;
            for (int j = 0; j < layers[1].neurons.size(); j++) {
                error += layers[1].deltas[j] * layers[1].weights[j][i];
            }
            layers[0].deltas[i] = error * activate_derivative(layers[0].neurons[i]);
        }

        // Update weights and biases
        for (int i = 0; i < layers[1].neurons.size(); i++) {
            for (int j = 0; j < layers[0].neurons.size(); j++) {
                layers[1].weights[i][j] += learning_rate * layers[1].deltas[i] * layers[0].neurons[j];
            }
            layers[1].biases[i] += learning_rate * layers[1].deltas[i];
        }

        for (int i = 0; i < layers[0].neurons.size(); i++) {
            for (int j = 0; j < input.size(); j++) {
                layers[0].weights[i][j] += learning_rate * layers[0].deltas[i] * input[j];
            }
            layers[0].biases[i] += learning_rate * layers[0].deltas[i];
        }
    }

    void train(const vector<vector<double> > & training_data, const vector<vector<double> > & labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_error = 0.0;

            for (size_t i = 0; i < training_data.size(); i++) {
                auto prediction = forward(training_data[i]);
                backward(training_data[i], labels[i]);

                for (size_t j = 0; j < prediction.size(); j++) {
                    total_error += pow(labels[i][j] - prediction[j], 2);
                }
            }

            if (epoch % 100 == 0) {
                cout << "Epoch " << epoch << ", Error: " << total_error / training_data.size() << endl;
            }
        }
    }
};

// Drug feature extractor class
class DrugFeatureExtractor {
public:
    static vector<double> extractFeatures(const string& smiles) {
        vector<double> features(100, 0.0);

        for (char c : smiles) {
            features[c % 100] += 1.0;
        }

        double sum = 0.0;
        for (double f : features) sum += f;
        if (sum > 0) {
            for (double& f : features) f /= sum;
        }

        return features;
    }
};

// Main drug classifier class
class DrugMechanismClassifier {
private:
    NeuralNetwork nn;
    vector<string> mechanism_classes;

public:

    DrugMechanismClassifier() 
        : nn(100, 64, 5) {
        mechanism_classes.push_back("Enzyme inhibitor");
        mechanism_classes.push_back("Receptor agonist");
        mechanism_classes.push_back("Receptor antagonist");
        mechanism_classes.push_back("Ion channel modulator");
        mechanism_classes.push_back("DNA intercalator");
    }

    void train(const vector<string>& smiles_data, const vector<int>& mechanisms, int epochs = 1000) {
        vector<vector<double> >  training_data;
        vector<vector<double> >  labels;

        for (size_t i = 0; i < smiles_data.size(); i++) {
            auto features = DrugFeatureExtractor::extractFeatures(smiles_data[i]);
            training_data.push_back(features);

            vector<double> label(mechanism_classes.size(), 0.0);
            label[mechanisms[i]] = 1.0;
            labels.push_back(label);
        }

        nn.train(training_data, labels, epochs);
    }

    string predict(const string& smiles) {
        auto features = DrugFeatureExtractor::extractFeatures(smiles);
        auto output = nn.forward(features);

        int max_idx = 0;
        for (size_t i = 1; i < output.size(); i++) {
            if (output[i] > output[max_idx]) max_idx = i;
        }

        return mechanism_classes[max_idx];
    }
};

// Load dataset function
void loadDataset(const string& filename, vector<string>& smiles_data, vector<int>& mechanisms) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    getline(file, line);
    while (getline(file, line)) {
        stringstream ss(line);
        string smiles, mechanism;

        if (getline(ss, smiles, ',') && getline(ss, mechanism)) {
            smiles_data.push_back(smiles);
            mechanisms.push_back(stoi(mechanism));
        }
    }

    file.close();
}

int main() {
    DrugMechanismClassifier classifier;

    vector<string> training_smiles;
    vector<int> training_mechanisms;

    string dataset_path = "synthetic_drug_dataset.csv"; // Update with the actual file path
    loadDataset(dataset_path, training_smiles, training_mechanisms);

    classifier.train(training_smiles, training_mechanisms, 1000);

    string test_smiles = "CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)CC(=O)NC3=CC=C(C=C3)F";
    string predicted_mechanism = classifier.predict(test_smiles);

    cout << "Predicted mechanism: " << predicted_mechanism << endl;

    return 0;
}