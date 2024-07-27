import hashlib
import re


def ensure_str(value: str | bytes) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, bytes):
        return value.decode("utf-8")
    else:
        raise ValueError("Input value must be str or bytes.")


def ensure_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        return value.encode("utf-8")
    else:
        raise ValueError("Input value must be str or bytes.")


class GlobalDictionary:
    def __init__(self) -> None:
        self.char_store: dict[bytes, str] = {}
        self.word_store: dict[bytes, str] = {}
        self.phrase_store: dict[bytes, str] = {}
        self.embedding_store: dict[bytes, list[float]] = {}

    def get_hash(self, text: str | bytes) -> bytes:
        """Generate a SHA-256 hash for the given text and return its bytes representation."""
        text = ensure_bytes(text)
        hash_object = hashlib.sha256(text).digest()
        return hash_object

    def add_char(self, char: str | bytes) -> bytes:
        """Add a character to the dictionary and return its hash."""
        hash_value = self.get_hash(char)
        char = ensure_str(char)
        if hash_value not in self.char_store:
            self.char_store[hash_value] = char
        return hash_value

    def add_word(self, word: str | bytes) -> bytes:
        """Add a word to the dictionary and return its hash."""
        word = ensure_str(word)
        char_hashes = [self.add_char(c) for c in word]
        combined_hash = self.get_hash(b"".join(char_hashes))
        if combined_hash not in self.word_store:
            self.word_store[combined_hash] = word
        return combined_hash

    def add_phrase(self, phrase: str | bytes) -> bytes:
        """Add a phrase or sentence to the dictionary and return its hash."""
        phrase = ensure_str(phrase)
        tokens = re.findall(r"\w+|[^\w\s]", phrase)
        token_hashes = [
            self.add_word(token) if token.isalnum() else self.add_char(token)
            for token in tokens
        ]
        combined_hash = self.get_hash(b"".join(token_hashes))
        if combined_hash not in self.phrase_store:
            self.phrase_store[combined_hash] = phrase
        return combined_hash

    def get_char(self, hash_value: bytes) -> str:
        """Retrieve the character for a given hash."""
        return self.char_store.get(hash_value, "Not found")

    def get_word(self, hash_value: bytes) -> str:
        """Retrieve the word for a given hash."""
        return self.word_store.get(hash_value, "Not found")

    def get_phrase(self, hash_value: bytes) -> str:
        """Retrieve the phrase for a given hash."""
        return self.phrase_store.get(hash_value, "Not found")

    def add_embedding(self, hash_value: bytes, embedding: list[float]) -> None:
        """Add an embedding for a given hash."""
        self.embedding_store[hash_value] = embedding

    def get_embedding(self, hash_value: bytes) -> list[float]:
        """Retrieve the embedding for a given hash."""
        return self.embedding_store.get(hash_value, [])
