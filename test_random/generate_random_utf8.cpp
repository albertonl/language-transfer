#include <iostream>
#include <fstream>
#include <cstdlib>
#include <map>

#define OUTPUT_FILE(size) "temp_mc4_garbage_train_" + size + ".bin"
#define STATS_FILE(size) "temp_mc4_garbage_train_" + size + ".stats"
#define MIN_SEQ_LENGTH 32
#define MAX_SEQ_LENGTH 1024
#define LOGGING_STEPS 5000LL

using namespace std;

uint8_t randomByte(uint8_t first, uint8_t last);
void writeStats(long long numSequences, long long byteLength, long long maxSize, const string SIZE);

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <DATASET_SIZE>\n"
             << "Allowed sizes: 6M, 19M, 60M, 189M, 600M, 6B" << endl;
        return 1;
    }

    const string SIZE = argv[1];

    unsigned long long totalLength = 0;
    unsigned long long numSequences = 0;
    uint32_t length;
    uint8_t* data = nullptr;
    ofstream out(OUTPUT_FILE(SIZE), ios::binary);

    const map<string, long long> sizes = {
        {"6M", 6815744LL},
        {"19M", 19398656LL},
        {"60M", 60817408LL},
        {"189M", 189267968LL},
        {"600M", 600834048LL},
        {"6B", 6001000448LL}
    };
    
    if (sizes.find(SIZE) == sizes.end()) {
        cerr << "Unknown size value \'" << SIZE << "\'" << endl;
        return 1;
    }

    if (!out) {
        cerr << "Problem opening output file" << endl;
        return 1;
    }

    srand(1234); // constant seed for reproducibility

    while (totalLength < sizes.at(SIZE)) {
        length = rand() % (MAX_SEQ_LENGTH + 1 - MIN_SEQ_LENGTH) + MIN_SEQ_LENGTH;
        totalLength += length;
        
        if (data == nullptr) {
            data = new uint8_t[length];
        } else {
            cerr << "Unhandled memory still in heap!" << endl;
            return 1;
        }

        for (unsigned i = 0; i < length; i++) {
            uint8_t first;
            do {
                first = randomByte(0x00, 0xF4);
            } while (first > 0x7F && first < 0xC2);

            if (first <= 0x7F) {
                data[i] = first;
            } else if (first <= 0xDF && i < length-1) {
                data[i] = first;
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first == 0xE0 && i < length-2) {
                data[i] = first;
                data[++i] = randomByte(0xA0, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first == 0xED && i < length-2) {
                data[i] = first;
                data[++i] = randomByte(0x80, 0x9F);
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first == 0xEF && i < length-2) {
                data[i] = first;
                data[++i] = randomByte(0x80, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first == 0xF0 && i < length-3) {
                data[i] = first;
                data[++i] = randomByte(0x90, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first <= 0xF3 && i < length-3) {
                data[i] = first;
                data[++i] = randomByte(0x80, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
            } else if (first == 0xF4 && i < length-3) {
                data[i] = first;
                data[++i] = randomByte(0x80, 0x8F);
                data[++i] = randomByte(0x80, 0xBF);
                data[++i] = randomByte(0x80, 0xBF);
            }
        }

        out.write(reinterpret_cast<const char*>(&length), sizeof(uint32_t));
        out.write((const char*) data, length);

        delete[] data;
        data = nullptr;

        if (++numSequences % LOGGING_STEPS == 0) {
            cout << "[" << totalLength << "/" << sizes.at(SIZE) << "] " << setprecision(2)
                 << 100.0 * ((double) totalLength / sizes.at(SIZE)) << "\% completed: "
                 << numSequences << " sequences" << endl;
        }
    }

    out.close();

    writeStats(numSequences, totalLength, sizes.at(SIZE), SIZE);

    return 0;
}

uint8_t randomByte(uint8_t first, uint8_t last) {
    return rand() % (last + 1 - first) + first;
}

void writeStats(long long numSequences, long long byteLength, long long maxSize, const string SIZE) {
    ofstream stats(STATS_FILE(SIZE));

    if (stats) {
        stats << "Dataset: " << OUTPUT_FILE(SIZE) << endl
            << "Number of sequences: " << numSequences << endl
            << "Total combined length of sequences (bytes): " << byteLength << endl
            << "Maximum expected combined length (bytes): " << maxSize << endl;
    }

    stats.close();
}