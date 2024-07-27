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


class Tokenizer:
    def __init__(self, global_dict: "GlobalDictionary"):
        self.global_dict = global_dict

    def tokenize_and_embed(self, text: str | bytes) -> list[bytes]:
        """
        Tokenize the input text into sentences, phrases, words, and characters,
        and embed them into the dictionary.
        Returns a list of hashes corresponding to the tokens.
        """
        text = ensure_str(text)
        sentence_tokens = re.split(r"(?<=[.!?]) +", text)  # Split text into sentences
        token_hashes = []

        for sentence in sentence_tokens:
            sentence_hash = self.global_dict.get_hash(sentence)
            if sentence_hash in self.global_dict.phrase_store:
                token_hashes.append(sentence_hash)
            else:
                # Break the sentence into phrases
                phrases = re.split(r"[,;]", sentence)
                phrase_hashes = []
                for phrase in phrases:
                    phrase_hash = self.global_dict.get_hash(phrase)
                    if phrase_hash in self.global_dict.phrase_store:
                        token_hashes.append(phrase_hash)
                    else:
                        phrase_hashes.append(phrase)

                # Process remaining phrases
                for phrase in phrase_hashes:
                    # Break the phrase into words
                    words = re.findall(r"\w+|[^\w\s]", phrase)
                    word_hashes = []
                    for word in words:
                        word_hash = self.global_dict.get_hash(word)
                        if word_hash in self.global_dict.word_store:
                            token_hashes.append(word_hash)
                        else:
                            word_hashes.append(word)

                    # Process remaining words
                    for word in word_hashes:
                        # Break the word into characters
                        char_hashes = []
                        for char in word:
                            char_hash = self.global_dict.get_hash(char)
                            if char_hash not in self.global_dict.char_store:
                                self.global_dict.add_char(char)
                            char_hashes.append(char_hash)
                        # After processing the characters, add the word to the dictionary
                        combined_word_hash = self.global_dict.get_hash(
                            b"".join(char_hashes)
                        )
                        self.global_dict.add_word(word)
                        token_hashes.append(combined_word_hash)

                    # After processing the words, add the phrase to the dictionary
                    combined_phrase_hash = self.global_dict.get_hash(
                        b"".join(token_hashes[-len(words) :])
                    )
                    self.global_dict.add_phrase(phrase)
                    token_hashes.append(combined_phrase_hash)

                # After processing the phrases, add the sentence to the dictionary
                combined_sentence_hash = self.global_dict.get_hash(
                    b"".join(token_hashes[-len(phrases) :])
                )
                self.global_dict.add_phrase(sentence)
                token_hashes.append(combined_sentence_hash)

        return token_hashes
