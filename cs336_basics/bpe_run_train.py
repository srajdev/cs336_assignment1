from bpe_trainer import train_bpe

if __name__ == "__main__":
    import cProfile
    # Initialize the BPE tokenizer
    #input_path = "/Users/hap-106/dev/cs336/assignment1-basics/tests/fixtures/corpus.en"
    PROJECT_ROOT = "/workspace" #"/Users/hap-106/dev"
    file_name = 'owt_train.txt' #"TinyStoriesV2-GPT4-train.txt"
    input_path =  PROJECT_ROOT + '/cs336/assignment1-basics/data/' + file_name
    vocab_size = 32000
    special_token = "<|endoftext|>"

    #cProfile.run("train_bpe(input_path, vocab_size, [special_token], 32)")
    vocab, merges = train_bpe(input_path, vocab_size, [special_token], num_processes=64)

    import pickle
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('merges.pkl', 'wb') as f:
        pickle.dump(merges, f)

    ## Save vocab and merges to files
    #import json

    # Save vocab as JSON, using string keys since JSON doesn't support int keys
    #with open('vocab.json', 'w') as f:
    #    # Convert bytes to list of ints for JSON serialization
    #    json_vocab = {str(k): list(v) for k,v in vocab.items()}
    #    json.dump(json_vocab, f)

    ## Save merges as JSON list of int lists
    #with open('merges.json', 'w') as f:
    #    # Convert bytes tuples to lists of ints
    #    json_merges = [list(m[0]) + list(m[1]) for m in merges]
    #    json.dump(json_merges, f)
    
    #tokenizer = BPETokenizer(input_path, vocab_size, [special_token])
    
    # Build the tokens
    #tokenizer.build_tokens()
    
    # Print results
    #print("\nVocabulary:")
    #print(len(tokenizer.vocab))
    
    #print("\nMerges:")
    #for i in range(len(tokenizer.merges)):
    #    print(f"{i}: {tokenizer.merges[i]}")
    
    #print("\nLast 30 vocabulary items:")
    #vocab_items = list(tokenizer.vocab.items())
    #for idx, item in vocab_items[-30:]:
    #    print(f"{idx}: {item}")


