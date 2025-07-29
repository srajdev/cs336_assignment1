# I need to create a BPE tokenizer that can tokenize a string into a list of tokens.
# <|endoftext|> is the special token for the end of the text.
# take the file, create chuncks.
# for each chunck run the regres that splits out the works along with white space
# create a tuple of tyhese usb tokenizert with their count
# then count the conscutive pairs in tuple and create a count of all the pairs
# find the hightest pair and merge it and create a new encoding
# keep doing till the vocab size is met, or do we go with # of merges??
# we have 256 + 1 (endoftext) tokens to start with
# return a list of all the tokens and their bytes
# a list of all the merges?


#from pretokenization_example import find_chunk_boundaries

import os
import regex as re
import time
import functools
from typing import List, Dict, Tuple, Optional, BinaryIO
from collections import Counter
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def time_function(func):
    """Decorator that tracks how long a function takes to execute."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        if func.__name__ in ["train_bpe"]:
            execution_time = end_time - start_time
            print(f"{func.__name__} took {execution_time:.6f} seconds to execute")
        return result
    return wrapper

class BPETokenizer:
    @time_function
    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]) -> None:
        """
        Initialize the BPETokenizer.

        Args:
            input_path (str): Path to the input text file.
            vocab_size (int): Desired vocabulary size for the tokenizer.
            special_tokens (List[str]): List of special tokens to use (e.g., end of text, padding, etc.).
        """
        self.input_path: str = input_path
        self.vocab_size: int = vocab_size
        self.special_tokens: List[str] = special_tokens
        self.vocab: Dict[int, bytes] = {}
        self.latest_vocab_index: int = 0
        self.merges: List[Tuple[bytes, bytes]] = []
        # Add persistent pair counts for optimization
        self.pair_counts: Dict[Tuple[bytes, bytes], int] = {}
        # NEW: Track which tokens contain each pair for faster updates
        self.pair_to_tokens: Dict[Tuple[bytes, bytes], set[Tuple[bytes, ...]]] = {}
        #self.load_vocab()
        #self.load_merges()

    @time_function
    def load_vocab(self) -> None:
        """
        Initialize the vocabulary with all single-byte values and the special tokens.
        """
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.latest_vocab_index = len(self.vocab)
        for special_token in self.special_tokens:
            self.vocab[self.latest_vocab_index] = special_token.encode("utf-8")
            self.latest_vocab_index += 1

    @time_function
    def load_documents(self, input_path: str) -> List[str]:
        """
        Load documents from the input file, splitting them by the first special token.

        Args:
            input_path (str): Path to the input text file.

        Returns:
            List[str]: List of documents split by the first special token.
        """
        documents = []
        with open(input_path, 'r') as file:
            content = file.read()
            documents = [content]
            for special_token in self.special_tokens:
                new_documents = []
                for doc in documents:
                    new_documents.extend(doc.split(special_token))
                documents = new_documents
        return documents

    @time_function
    def _count_chunk(self, content: str) -> Dict[Tuple[bytes, ...], int]:
        # Split the content into documents
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""                        
        documents = [content]
        for special_token in self.special_tokens:
            new_documents = []
            for doc in documents:
                new_documents.extend(doc.split(special_token))
            documents = new_documents

        counts: Dict[Tuple[bytes, ...], int] = {}
        for d in documents:
            for match in re.finditer(PAT, d):
                token = match.group()
                token_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                counts[token_tuple] = counts.get(token_tuple, 0) + 1
        return counts
    
    @time_function
    def build_count_list(self, input_path: str, num_processes: Optional[int] = None) -> Dict[Tuple[bytes, ...], int]:
        """
        Build a count dictionary of token tuples from the loaded documents.

        Args:
            input_path (str): Path to the input text file (not used, relies on self.documents).
            num_processes (Optional[int]): Number of processes to use for parallel processing. If None, uses serial processing.
        
        Returns:
            Dict[Tuple[bytes, ...], int]: Dictionary mapping token tuples to their frequency count.
        """
        if not num_processes:
            content = None
            with open(input_path, 'r') as file:
                content = file.read()
            counts = self._count_chunk(content)
            return counts
        
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, self.special_tokens[0].encode("utf-8"))
            
            chunk_params = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_params.append((chunk,))
                    
            with Pool(num_processes) as pool:
                chunk_counts = pool.starmap(self._count_chunk, chunk_params)
                
            # Merge all chunk counts                
            final_count = result = sum((Counter(d) for d in chunk_counts), Counter())
            final_counts: Dict[Tuple[bytes, ...], int] = dict(final_count)
            return final_counts

    
    @time_function
    def initialize_pair_counts(self, token_counts: Dict[Tuple[bytes, ...], int]) -> None:
        """
        Initialize the pair counts dictionary from token counts. This is called once.

        Args:
            token_counts (Dict[Tuple[bytes, ...], int]): Dictionary of token tuples and their counts.
        """
        self.pair_counts = {}
        self.pair_to_tokens = {} # Clear previous data
        for k, v in token_counts.items():
            for i in range(len(k) - 1):
                pair = (k[i], k[i+1])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + v
                # Add tokens containing this pair to the set
                if pair in self.pair_to_tokens:
                    self.pair_to_tokens[pair].add(k)
                else:
                    self.pair_to_tokens[pair] = {k}
    
    @time_function
    def find_most_frequent_pair(self) -> Tuple[bytes, bytes]:
        """
        Find and return the most frequent pair from the current pair counts.
        
        Returns:
            Tuple[bytes, bytes]: The most frequent pair.
        """
        if not self.pair_counts:
            raise ValueError("No pairs available to merge")
        
        # Find the pair with maximum count, using lexicographic order as tiebreaker
        return max(self.pair_counts, key=lambda k: (self.pair_counts[k], k))
    
    #@time_function
    def merge_tuple_with_match(self, token_tuple: Tuple[bytes, ...], best_tuple: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
        """
        Merge consecutive bytes in a token tuple that match the given best pair.

        Args:
            token_tuple (Tuple[bytes, ...]): The tuple of bytes to process.
            best_tuple (Tuple[bytes, bytes]): The pair of bytes to merge when found.

        Returns:
            Tuple[bytes, ...]: A new tuple with matching pairs merged into single bytes.
        """
        if len(token_tuple) < 2:
            return token_tuple
        
        result: List[bytes] = []
        i: int = 0
        while i < len(token_tuple):
            if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == best_tuple:
                merged = token_tuple[i] + token_tuple[i+1]
                result.append(merged)
                i += 2
            else:
                result.append(token_tuple[i])
                i += 1

        return tuple(result)

    @time_function
    def update_pair_counts(self, token_counts: Dict[Tuple[bytes, ...], int], merge_pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
        """
        Update pair counts incrementally when performing a merge.
        
        Args:
            token_counts: Current token counts
            merge_pair: The pair being merged
            
        Returns:
            Updated token counts
        """
        updated_counts = {}
        new_token = merge_pair[0] + merge_pair[1]
        
        # Get all tokens that contain the pair being merged
        tokens_to_update = self.pair_to_tokens.get(merge_pair, set())
        
        # OPTIMIZATION: Only process tokens that might be affected
        for token_tuple, count in token_counts.items():
            if token_tuple in tokens_to_update:
                # This token contains the merge pair, so process it
                old_tuple = token_tuple
                new_tuple = self.merge_tuple_with_match(token_tuple, merge_pair)
                updated_counts[new_tuple] = count
                
                # Update pair counts since the tuple changed
                # Remove old pairs from this token
                for i in range(len(old_tuple) - 1):
                    old_pair = (old_tuple[i], old_tuple[i+1])
                    self.pair_counts[old_pair] -= count
                    if self.pair_counts[old_pair] <= 0:
                        del self.pair_counts[old_pair]
                    # Remove tokens containing the old pair
                    if old_pair in self.pair_to_tokens:
                        self.pair_to_tokens[old_pair].discard(old_tuple)
                        if not self.pair_to_tokens[old_pair]:
                            del self.pair_to_tokens[old_pair]
                
                # Add new pairs from the updated token
                for i in range(len(new_tuple) - 1):
                    new_pair = (new_tuple[i], new_tuple[i+1])
                    self.pair_counts[new_pair] = self.pair_counts.get(new_pair, 0) + count
                    # Add tokens containing the new pair
                    if new_pair in self.pair_to_tokens:
                        self.pair_to_tokens[new_pair].add(new_tuple)
                    else:
                        self.pair_to_tokens[new_pair] = {new_tuple}
            else:
                # Token is unchanged, just copy it over
                updated_counts[token_tuple] = count
        
        return updated_counts

    @time_function
    def perform_merge(self, token_counts: Dict[Tuple[bytes, ...], int], merge_pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
        """
        Perform a merge operation and update all data structures.
        
        Args:
            token_counts: Current token counts
            merge_pair: The pair to merge
            
        Returns:
            Updated token counts
        """
        # Update token counts and pair counts incrementally
        updated_counts = self.update_pair_counts(token_counts, merge_pair)
        
        # Add new token to vocabulary
        self.vocab[self.latest_vocab_index] = merge_pair[0] + merge_pair[1]
        self.latest_vocab_index += 1
        self.merges.append(merge_pair)
        
        return updated_counts

    @time_function
    def build_tokens(self, num_processes: Optional[int] = None):
        self.load_vocab()
        num_merges = self.vocab_size - len(self.vocab)
        token_counts = self.build_count_list(self.input_path, num_processes)
        
        # Initialize pair counts once
        self.initialize_pair_counts(token_counts)
        
        for i in range(num_merges):
            if not self.pair_counts:  # No more pairs to merge
                break
                
            # Find most frequent pair (now O(1) lookup instead of O(n) recalculation)
            merge_pair = self.find_most_frequent_pair()
            
            # Perform merge and update counts incrementally
            token_counts = self.perform_merge(token_counts, merge_pair)

@time_function
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: Optional[int] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    tokenizer.build_tokens(num_processes=num_processes)
    return (tokenizer.vocab, tokenizer.merges)  

if __name__ == "__main__":
    import cProfile
    # Initialize the BPE tokenizer
    #input_path = "/Users/hap-106/dev/cs336/assignment1-basics/tests/fixtures/corpus.en"
    PROJECT_ROOT = "/workspace" #"/Users/hap-106/dev"
    file_name = "TinyStoriesV2-GPT4-train.txt"
    input_path =  PROJECT_ROOT + '/cs336/assignment1-basics/data/' + file_name
    vocab_size = 10000
    special_token = "<|endoftext|>"

    #cProfile.run("train_bpe(input_path, vocab_size, [special_token], 32)")
    vocab, merges = train_bpe(input_path, vocab_size, [special_token], num_processes=32)

    ## Save vocab and merges to files
    #import json

    ## Save vocab as JSON, using string keys since JSON doesn't support int keys
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


