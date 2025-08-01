import os
import regex as re
import time
import functools
import pickle
import gzip
import json
from typing import List, Dict, Tuple, Optional, BinaryIO
from collections import Counter
from multiprocessing import Pool
import hashlib


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
        execution_time = end_time - start_time
        #print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper


# NEW: Worker function for multiprocessing (must be at module level)
def count_chunk_worker(chunk_data: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    """Worker function for counting tokens in a chunk - can be pickled."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Split the content into documents
    documents = [chunk_data]
    for special_token in special_tokens:
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


class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str], 
                 checkpoint_dir: str = "checkpoints", checkpoint_every: int = 100) -> None:
        """
        Initialize the BPETokenizer with checkpointing support.
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: Dict[int, bytes] = {}
        self.latest_vocab_index = 0
        self.merges: List[Tuple[bytes, bytes]] = []
        self.pair_counts: Dict[Tuple[bytes, bytes], int] = {}
        self.pair_to_tokens: Dict[Tuple[bytes, bytes], set[Tuple[bytes, ...]]] = {}
        
        # NEW: Checkpointing parameters
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.current_merge_step = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate file hash for cache validation
        self.file_hash = self._generate_file_hash()

    def _generate_file_hash(self) -> str:
        """Generate hash of input file for cache validation."""
        hasher = hashlib.md5()
        with open(self.input_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]  # Use first 8 chars

    def _get_checkpoint_path(self, step: int) -> str:
        """Get checkpoint file path for a given step."""
        base_name = os.path.basename(self.input_path).split('.')[0]
        return os.path.join(
            self.checkpoint_dir, 
            f"bpe_{base_name}_{self.vocab_size}_{self.file_hash}_step_{step}.pkl.gz"
        )

    def _get_token_counts_cache_path(self) -> str:
        """Get cache path for initial token counts."""
        base_name = os.path.basename(self.input_path).split('.')[0]
        return os.path.join(
            self.checkpoint_dir,
            f"token_counts_{base_name}_{self.file_hash}.pkl.gz"
        )

    def save_checkpoint(self, step: int, token_counts: Dict[Tuple[bytes, ...], int]) -> None:
        """Save current training state to checkpoint."""
        checkpoint_data = {
            'vocab': self.vocab,
            'latest_vocab_index': self.latest_vocab_index,
            'merges': self.merges,
            'pair_counts': self.pair_counts,
            'pair_to_tokens': self.pair_to_tokens,
            'token_counts': token_counts,
            'current_merge_step': step,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'file_hash': self.file_hash
        }
        
        checkpoint_path = self._get_checkpoint_path(step)
        with gzip.open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Checkpoint saved at step {step}: {checkpoint_path}")

    def load_checkpoint(self, step: int = None) -> Tuple[Optional[int], Optional[Dict[Tuple[bytes, ...], int]]]:
        """Load checkpoint from file. If step is None, loads the latest checkpoint."""
        if step is None:
            # Find latest checkpoint
            checkpoint_files = []
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith(f"bpe_") and f.endswith(".pkl.gz") and self.file_hash in f:
                    try:
                        step_num = int(f.split('_step_')[1].split('.')[0])
                        checkpoint_files.append((step_num, f))
                    except (IndexError, ValueError):
                        continue
            
            if not checkpoint_files:
                return None, None
            
            # Get latest checkpoint
            step, filename = max(checkpoint_files)
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        else:
            checkpoint_path = self._get_checkpoint_path(step)
            if not os.path.exists(checkpoint_path):
                return None, None

        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint compatibility
            if (checkpoint_data['vocab_size'] != self.vocab_size or 
                checkpoint_data['special_tokens'] != self.special_tokens or
                checkpoint_data['file_hash'] != self.file_hash):
                print("Checkpoint incompatible with current configuration")
                return None, None
            
            # Restore state
            self.vocab = checkpoint_data['vocab']
            self.latest_vocab_index = checkpoint_data['latest_vocab_index']
            self.merges = checkpoint_data['merges']
            self.pair_counts = checkpoint_data['pair_counts']
            self.pair_to_tokens = checkpoint_data['pair_to_tokens']
            self.current_merge_step = checkpoint_data['current_merge_step']
            
            print(f"Loaded checkpoint from step {self.current_merge_step}")
            return self.current_merge_step, checkpoint_data['token_counts']
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None

    def cache_token_counts(self, token_counts: Dict[Tuple[bytes, ...], int]) -> None:
        """Cache the initial token counts."""
        cache_path = self._get_token_counts_cache_path()
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(token_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Token counts cached: {cache_path}")

    def load_cached_token_counts(self) -> Optional[Dict[Tuple[bytes, ...], int]]:
        """Load cached token counts if available."""
        cache_path = self._get_token_counts_cache_path()
        if os.path.exists(cache_path):
            try:
                with gzip.open(cache_path, 'rb') as f:
                    token_counts = pickle.load(f)
                print(f"Loaded cached token counts: {cache_path}")
                return token_counts
            except Exception as e:
                print(f"Error loading cached token counts: {e}")
        return None

    @time_function
    def load_vocab(self) -> None:
        """Initialize the vocabulary with all single-byte values and the special tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.latest_vocab_index = len(self.vocab)
        for special_token in self.special_tokens:
            self.vocab[self.latest_vocab_index] = special_token.encode("utf-8")
            self.latest_vocab_index += 1

    @time_function
    def build_count_list(self, input_path: str, num_processes: Optional[int] = None) -> Dict[Tuple[bytes, ...], int]:
        """Build a count dictionary of token tuples from the loaded documents."""
        # Try to load from cache first
        cached_counts = self.load_cached_token_counts()
        if cached_counts is not None:
            return cached_counts

        print(f"Building token counts with {num_processes or 1} processes...")
        
        if not num_processes:
            content = None
            with open(input_path, 'r') as file:
                content = file.read()
            counts = count_chunk_worker(content, self.special_tokens)
            self.cache_token_counts(counts)
            return counts
        
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, self.special_tokens[0].encode("utf-8"))
            
            chunk_params = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_params.append((chunk, self.special_tokens))
                    
            with Pool(num_processes) as pool:
                chunk_counts = pool.starmap(count_chunk_worker, chunk_params)
                
            # Merge all chunk counts                
            final_count = sum((Counter(d) for d in chunk_counts), Counter())
            final_counts: Dict[Tuple[bytes, ...], int] = dict(final_count)
            
            # Cache the results
            self.cache_token_counts(final_counts)
            return final_counts

    @time_function
    def initialize_pair_counts(self, token_counts: Dict[Tuple[bytes, ...], int]) -> None:
        """Initialize the pair counts dictionary from token counts."""
        print("Initializing pair counts...")
        self.pair_counts = {}
        self.pair_to_tokens = {}
        
        for k, v in token_counts.items():
            for i in range(len(k) - 1):
                pair = (k[i], k[i+1])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + v
                if pair in self.pair_to_tokens:
                    self.pair_to_tokens[pair].add(k)
                else:
                    self.pair_to_tokens[pair] = {k}

    def find_most_frequent_pair(self) -> Tuple[bytes, bytes]:
        """Find and return the most frequent pair from the current pair counts."""
        if not self.pair_counts:
            raise ValueError("No pairs available to merge")
        
        return max(self.pair_counts, key=lambda k: (self.pair_counts[k], k))

    @time_function
    def merge_tuple_with_match(self, token_tuple: Tuple[bytes, ...], best_tuple: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
        """Merge consecutive bytes in a token tuple that match the given best pair."""
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
        """Start with copy, then modify only what's needed."""
        start_time = time.perf_counter()
        
        # FAST: Copy the entire dictionary at once
        copy_start = time.perf_counter()
        updated_counts = token_counts.copy()  # O(n) but very fast in C
        copy_time = time.perf_counter() - copy_start
        
        # FIX: Make a copy of the set to avoid "set changed size during iteration"
        tokens_to_update = self.pair_to_tokens.get(merge_pair, set()).copy()
        
        # FAST: Only process the small subset that needs changes
        update_start = time.perf_counter()
        for token_tuple in tokens_to_update:
            if token_tuple in updated_counts:  # Make sure it still exists
                count = updated_counts[token_tuple]
                
                # Remove old token
                del updated_counts[token_tuple]
                
                # Add updated token
                old_tuple = token_tuple
                new_tuple = self.merge_tuple_with_match(token_tuple, merge_pair)
                updated_counts[new_tuple] = count
                
                # Update pair counts
                for i in range(len(old_tuple) - 1):
                    old_pair = (old_tuple[i], old_tuple[i+1])
                    self.pair_counts[old_pair] -= count
                    if self.pair_counts[old_pair] <= 0:
                        del self.pair_counts[old_pair]
                    if old_pair in self.pair_to_tokens:
                        self.pair_to_tokens[old_pair].discard(old_tuple)
                        if not self.pair_to_tokens[old_pair]:
                            del self.pair_to_tokens[old_pair]
                
                # Add new pairs
                for i in range(len(new_tuple) - 1):
                    new_pair = (new_tuple[i], new_tuple[i+1])
                    self.pair_counts[new_pair] = self.pair_counts.get(new_pair, 0) + count
                    if new_pair in self.pair_to_tokens:
                        self.pair_to_tokens[new_pair].add(new_tuple)
                    else:
                        self.pair_to_tokens[new_pair] = {new_tuple}
        
        update_time = time.perf_counter() - update_start
        total_time = time.perf_counter() - start_time
        
        #if total_time > 0.5:
        #    print(f"Copy method: {total_time:.3f}s (copy: {copy_time:.3f}s, update: {update_time:.3f}s)")
        #    print(f"Tokens updated: {len(tokens_to_update):,}/{len(token_counts):,} ({len(tokens_to_update)/len(token_counts)*100:.2f}%)")
        
        return updated_counts
    

#    def update_pair_counts(self, token_counts: Dict[Tuple[bytes, ...], int], merge_pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
#        """Update pair counts incrementally when performing a merge."""
#        updated_counts = {}
#        
#        # Get all tokens that contain the pair being merged
#        tokens_to_update = self.pair_to_tokens.get(merge_pair, set())
#        
#        for token_tuple, count in token_counts.items():
#            if token_tuple in tokens_to_update:
#                old_tuple = token_tuple
#                new_tuple = self.merge_tuple_with_match(token_tuple, merge_pair)
#                updated_counts[new_tuple] = count
#                
#                # Update pair counts
#                for i in range(len(old_tuple) - 1):
#                    old_pair = (old_tuple[i], old_tuple[i+1])
#                    self.pair_counts[old_pair] -= count
#                    if self.pair_counts[old_pair] <= 0:
#                        del self.pair_counts[old_pair]
#                    if old_pair in self.pair_to_tokens:
#                        self.pair_to_tokens[old_pair].discard(old_tuple)
#                        if not self.pair_to_tokens[old_pair]:
#                            del self.pair_to_tokens[old_pair]
#                
#                # Add new pairs
#                for i in range(len(new_tuple) - 1):
#                    new_pair = (new_tuple[i], new_tuple[i+1])
#                    self.pair_counts[new_pair] = self.pair_counts.get(new_pair, 0) + count
#                    if new_pair in self.pair_to_tokens:
#                        self.pair_to_tokens[new_pair].add(new_tuple)
#                    else:
#                        self.pair_to_tokens[new_pair] = {new_tuple}
#            else:
#                updated_counts[token_tuple] = count
#        
#        return updated_counts
    
    @time_function
    def perform_merge(self, token_counts: Dict[Tuple[bytes, ...], int], merge_pair: Tuple[bytes, bytes]) -> Dict[Tuple[bytes, ...], int]:
        """Perform a merge operation and update all data structures."""
        updated_counts = self.update_pair_counts(token_counts, merge_pair)
        
        # Add new token to vocabulary
        self.vocab[self.latest_vocab_index] = merge_pair[0] + merge_pair[1]
        self.latest_vocab_index += 1
        self.merges.append(merge_pair)
        
        return updated_counts

    def save_final_results(self, output_dir: str = "output") -> None:
        """Save final vocabulary and merges to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(self.input_path).split('.')[0]
        
        # Save as pickle (most efficient)
        with gzip.open(os.path.join(output_dir, f"{base_name}_vocab_{self.vocab_size}.pkl.gz"), 'wb') as f:
            pickle.dump({'vocab': self.vocab, 'merges': self.merges}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save as JSON for inspection
        with open(os.path.join(output_dir, f"{base_name}_vocab_{self.vocab_size}.json"), 'w') as f:
            json_data = {
                'vocab': {str(k): list(v) for k, v in self.vocab.items()},
                'merges': [[list(m[0]), list(m[1])] for m in self.merges],
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to {output_dir}")

    @time_function
    def build_tokens(self, num_processes: Optional[int] = None, resume: bool = True):
        """Build tokens with checkpointing support."""
        # Try to resume from checkpoint
        if resume:
            step, token_counts = self.load_checkpoint()
            if step is not None and token_counts is not None:
                print(f"Resuming from step {step}")
                start_step = step + 1
            else:
                print("No compatible checkpoint found, starting fresh")
                self.load_vocab()
                token_counts = self.build_count_list(self.input_path, num_processes)
                self.initialize_pair_counts(token_counts)
                start_step = 0
        else:
            print("Starting fresh (ignoring checkpoints)")
            self.load_vocab()
            token_counts = self.build_count_list(self.input_path, num_processes)
            self.initialize_pair_counts(token_counts)
            start_step = 0

        num_merges = self.vocab_size - len(self.vocab)
        print(f"Need to perform {num_merges} more merges")

        end_step = start_step + num_merges
        
        for i in range(start_step, end_step):
            if not self.pair_counts:
                print("No more pairs to merge")
                break
                
            merge_pair = self.find_most_frequent_pair()
            token_counts = self.perform_merge(token_counts, merge_pair)
            self.current_merge_step = i
            
            # Progress tracking
            if i % 10 == 0:
                print(f"Merge {i}/{end_step}: {merge_pair[0]} + {merge_pair[1]} "
                      f"(count: {self.pair_counts.get(merge_pair, 0)})")
            
            # Checkpointing
            if i % self.checkpoint_every == 0:
                self.save_checkpoint(i, token_counts)
        
        # Final checkpoint
        self.save_checkpoint(self.current_merge_step, token_counts)
        self.save_final_results()


@time_function
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], 
              num_processes: Optional[int] = None, resume: bool = True,
              checkpoint_every: int = 100) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train BPE with checkpointing support."""
    tokenizer = BPETokenizer(input_path, vocab_size, special_tokens, 
                           checkpoint_every=checkpoint_every)
    tokenizer.build_tokens(num_processes=num_processes, resume=resume)
    return (tokenizer.vocab, tokenizer.merges)


if __name__ == "__main__":
    PROJECT_ROOT = "/workspace"
    file_name = "owt_train.txt"#"TinyStoriesV2-GPT4-valid.txt" #"TinyStoriesV2-GPT4-train.txt"
    input_path = PROJECT_ROOT + '/cs336_assignment1/data/' + file_name
    vocab_size = 32000
    special_token = "<|endoftext|>"

    vocab, merges = train_bpe(
        input_path, vocab_size, [special_token], 
        num_processes=4, resume=True, checkpoint_every=1000
    )
