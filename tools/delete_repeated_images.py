import os
import hashlib

def hash_file(path, block_size=65536):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buf = f.read(block_size)
        while buf:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def remove_duplicates(folder):
    seen = {}
    removed = 0

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if not os.path.isfile(filepath):
            continue

        # calcula hash
        file_hash = hash_file(filepath)

        if file_hash in seen:
            print(f"Duplicado encontrado: {filepath} (mantendo {seen[file_hash]})")
            os.remove(filepath)
            removed += 1
        else:
            seen[file_hash] = filepath

    print(f"\nRemovidos {removed} arquivos duplicados.")

if __name__ == "__main__":
    # pasta = "./imagens"  # altere para a pasta desejada
    # pasta = "/home/ale/Downloads/spaghetti"
    pasta = "/home/ale/Downloads/stringing"
    # pasta = "/home/ale/Downloads/PI2/3d_print_failure_dataset/train/images (cópia)"

    remove_duplicates(pasta)