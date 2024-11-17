#include <iostream>
#include <fstream>
#include <cstdlib>

#define OUTPUT_FILE "mc4_garbage_train_6M.bin"
#define STATS_FILE "mc4_garbage_train_6M.stats"
#define MAX_SIZE 6815744 //6001000448LL // 6B tokens (bytes)
#define MIN_SEQ_LENGTH 32
#define MAX_SEQ_LENGTH 1024
#define LOGGING_STEPS 5000LL

using namespace std;

uint8_t randomByte(uint8_t first, uint8_t last);
void writeStats(long long numSequences, long long byteLength);

int main() {
    unsigned long long totalLength = 0;
    unsigned long long numSequences = 0;
    uint32_t length;
    uint8_t* data = nullptr;
    ofstream out(OUTPUT_FILE, ios::binary);
    
    srand(1234); // constant seed for reproducibility

    if (!out) {
        cerr << "Problem opening output file" << endl;
        return 1;
    }

    while (totalLength < MAX_SIZE) {
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
            cout << "[" << totalLength << "/" << MAX_SIZE << "] " << setprecision(2)
                 << 100.0 * ((double) totalLength / MAX_SIZE) << "\% completed: "
                 << numSequences << " sequences" << endl;
        }
    }

    out.close();

    writeStats(numSequences, totalLength);

    return 0;
}

uint8_t randomByte(uint8_t first, uint8_t last) {
    return rand() % (last + 1 - first) + first;
}

void writeStats(long long numSequences, long long byteLength) {
    ofstream stats(STATS_FILE);

    if (stats) {
        stats << "Dataset: " << OUTPUT_FILE << endl
            << "Number of sequences: " << numSequences << endl
            << "Total combined length of sequences (bytes): " << byteLength << endl
            << "Maximum expected combined length (bytes): " << MAX_SIZE << endl;
    }

    stats.close();
}