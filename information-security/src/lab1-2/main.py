import logging
import argparse
import sys
from encryption.algos import ECB, CBC, CFB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main",
        description="Lab work for information security - block cyphers",
        epilog="ECB, CBC, CFB"
    )
    parser.add_argument("-i", "--input-file", dest="input_file")
    parser.add_argument("-o", "--output-file", dest="output_file")
    parser.add_argument("-k", "--key-file", dest="key_file")
    parser.add_argument("-a", "--algo", dest="algo", choices=("ECB", "CBC", "CFB"))
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--encode", action="store_true")
    action_group.add_argument("--decode", action="store_true")
    parser.add_argument("--initial-vector", dest="initial_vector", required=False)
    parser.add_argument("-l", "--log", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.log else logging.INFO
    logging.basicConfig(filename=".log", filemode="w", level=log_level)

    print(args)

    key_path = args.key_file
    with open(key_path, "rb") as key_file:
        key = key_file.read()
        key = int.from_bytes(key)

    if args.initial_vector:
        iv_path = args.initial_vector
        with open(iv_path, "rb") as iv_file:
            iv = iv_file.read()
            iv = int.from_bytes(iv)

    encoding = args.encode

    target_path = args.input_file
    with open(target_path, "rb") as target_file:
        byte_data = target_file.read()

    match args.algo:
        case "ECB":
            coder = ECB(key, 8)
        case "CBC":
            coder = CBC(key, key, 8)
        case "CFB":
            coder = CFB(key, key, 8)
        case _:
            print(f"Unsupported algo {args.algo}")
            sys.exit(1)

    result = coder.encode(byte_data) if encoding else coder.decode(byte_data)

    output_file_name = args.output_file
    with open(output_file_name, "wb") as output_file:
        output_file.write(result)
