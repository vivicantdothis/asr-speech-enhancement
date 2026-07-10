from src.enhancement.factory import build_enhancer
def main():
    enhancer=build_enhancer("deepfilternet")
    print(type(enhancer))
    print(enhancer.input_type)
if __name__=="__main__":
    main()