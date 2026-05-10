import random

def sample_corpus(input_path, output_path, sample_size=10000):
    """
    Sample a subset of the corpus and save it to a new file.
    :param input_path: Path to the original corpus file.
    :param output_path: Path to the new sampled file.
    :param sample_size: Number of samples to keep.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
        # Sample randomly from the lines (sample_size number of lines)
        sampled_lines = random.sample(lines, sample_size)
    
    # Write the sampled lines to a new file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(sampled_lines)
    
    print(f"Sampled {sample_size} lines and saved to {output_path}")

if __name__ == "__main__":
    # Adjust paths and sample size as needed
    sample_corpus("tokenizer/corpus.txt", "tokenizer/sampled_corpus.txt", sample_size=10000)

# run
# python src/tokenizer/sample_token_corpus.py