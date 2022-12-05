#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

typedef struct {
    int i_exp;
    int i_node;
    int outcome;
    double value;
    int is_perturbation;
}Input;

int main() {
    std::ifstream f("data.csv", std::ifstream::in);
    char delim =',';
    string header;
    getline(f, header);
    while(f) {
        Input input;
        string line;
        if (getline(f, line)) {
            // cout<< line << endl;
            istringstream ss(line);
            string token;
            getline(ss, token, delim);
            getline(ss, token, delim);
            input.i_exp = stoi(token);
            // cout << "iexp " << input.i_exp << endl;
            getline(ss, token, delim);
            input.i_node = stoi(token);
            getline(ss, token, delim);
            input.outcome = stoi(token);
            getline(ss, token, delim);
            input.value = stod(token);
            getline(ss, token, delim);
            input.is_perturbation = stoi(token);
            cout << input.i_exp << " " << input.i_node << " " << input.outcome << " " << input.value << " " << input.is_perturbation << endl;
        }
    }

}