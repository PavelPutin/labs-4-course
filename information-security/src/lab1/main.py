import logging
import sys
from encryption.algos import ECB

if __name__ == "__main__":
    logging.basicConfig(filename=".log", filemode="w", level=logging.DEBUG)
    key_path = sys.argv[1]
    with open(key_path, "rb") as key_file:
        K = key_file.read()
        K = int.from_bytes(K)
    encoding = sys.argv[2] == "encode"
    target_path = sys.argv[3]
    with open(target_path, "rb") as target_file:
        byte_data = target_file.read()

    ecb = ECB(K, 8)
    result = ecb.encode(byte_data) if encoding else ecb.decode(byte_data)

    output_file_name = target_path + ".encoded" if encoding else target_path + ".decoded"
    with open(output_file_name, "wb") as output_file:
        output_file.write(result)
