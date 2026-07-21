from pystoi import stoi

def compute_stoi(clean,enhanced,sr):
    return stoi(clean,enhanced,sr,extended=False)