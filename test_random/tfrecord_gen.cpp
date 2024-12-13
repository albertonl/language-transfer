#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <map>

using namespace std;

typedef struct tfrecord {
    uint64_t length;
    uint32_t masked_crc32_of_length;
    char* data;
    uint32_t masked_crc32_of_data;
} TFRecord;

void writeStats(char* filename, long long numSequences, long long totalLength, const string size);

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file.tfrecord>" << endl;
        return 1;
    }

    ifstream in(argv[1], ios::binary);
    ofstream out(strcat(argv[1], ".bin"), ios::binary);

    TFRecord record;

    long long numSequences = 0;
    long long totalLength = 0;
    string selectedSize;

    cout << "Introduce the dataset size [6M, 19M, 60M, 189M, 600M, 6B]: ";
    cin >> selectedSize;

    if (in && out) {
        while (!in.eof()) {
            // Read input TFRecord file, record by record.
            in.read((char*)&record.length, sizeof(uint64_t));
            in.read((char*)&record.masked_crc32_of_length, sizeof(uint32_t));

            record.data = new char[record.length];

            in.read(record.data, record.length);
            in.red((char*)&record.masked_crc32_of_data, sizeof(uint32_t));

            // Write output dataset file
            out.write(reinterpret_cast<const char*>(&record.length), sizeof(uint32_t));
            out.write((const char*) record.data, record.length);

            numSequences++;
            totalLength += record.length;

            delete[] record.data;
            record.data = nullptr;
        }
    }

    in.close();
    out.close();

    writeStats(argv[1], numSequences, totalLength, selectedSize);

    return 0;
}

void writeStats(char* filename, long long numSequences, long long totalLength, const string size) {
    ofstream stats(strcat(filename, ".stats"));

    const map<string, long long> sizes = {
        {"6M", 6815744LL},
        {"19M", 19398656LL},
        {"60M", 60817408LL},
        {"189M", 189267968LL},
        {"600M", 600834048LL},
        {"6B", 6001000448LL}
    };

    if (stats) {
        stats << "Dataset: " << strcat(filename, ".bin") << endl
              << "Number of sequences: " << numSequences << endl
              << "Total combined length of sequences (bytes): " << totalLength << endl
              << "Maximum expected combined length (bytes): " << sizes.at(size) << endl;
    }

    stats.close();
}