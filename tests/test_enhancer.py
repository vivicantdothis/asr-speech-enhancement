import torch

from src.enhancement import build_enhancer
def main():
    enhancer=build_enhancer("cnn")
    x=torch.randn(1,1,80,300)
    y=enhancer.enhance(x)
    print("Input:",x.shape)
    print("Output:",y.shape)

if __name__=="__main__":
    main()