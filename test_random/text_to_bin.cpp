#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>

#define EXTRACTION_SIZE 10

using namespace std;

typedef struct tfrecord {
    uint64_t length;
    uint32_t masked_crc32_of_length;
    char* data;
    uint32_t masked_crc32_of_data;
} TFRecord;

int main(int argc, const char* argv[]) {
    ifstream in("datasets/mc4_es_train_6M.tfrecord", ios::binary);
    vector<TFRecord> records(EXTRACTION_SIZE);

    for (int i = 0; i < EXTRACTION_SIZE; i++) {
        in.read((char*)&records[i].length, sizeof(uint64_t));
        in.read((char*)&records[i].masked_crc32_of_length, sizeof(uint32_t));

        records[i].data = new char[records[i].length];

        in.read(records[i].data, records[i].length);
        in.read((char*)&records[i].masked_crc32_of_data, sizeof(uint32_t));
    }

    for (int i = 0; i < min(10LL, (long long) records[0].length); i++) {
        bitset<8> bits(records[0].data[i]);
        cout << "Character no. " << i+1 << " = " << bits << endl;
    }

    for (int i = 0; i < records.size(); i++) {
        delete[] records[i].data;
    }

    in.close();

    return 0;
}