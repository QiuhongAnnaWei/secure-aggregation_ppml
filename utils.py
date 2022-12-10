import pickle
import os
from safe_primes import PRIMES
from secretsharing import SecretSharer
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHA256

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Diffie-Hellman Kuba Wrapper created by Kuba Szczodrzyński
class DH:
    _prime: int
    _private_key: int
    _public_key: int
    _shared_key: int

    @staticmethod
    def _to_bytes(a: int) -> bytes:
        return a.to_bytes((a.bit_length() + 7) // 8, byteorder="big")

    def __init__(self, group: int = 14, key_bits: int = 540) -> None:
        prime_bytes = PRIMES[group]
        self._prime = int.from_bytes(prime_bytes, byteorder="big")
        self.generate_private_key(key_bits)

    def generate_private_key(self, key_bits: int = 540) -> bytes:
        private_key = os.urandom(key_bits // 8 + 8)
        self.set_private_key(private_key)
        return self.get_private_key()

    def set_private_key(self, key: bytes) -> None:
        self._private_key = int.from_bytes(key, byteorder="big")
        self._public_key = pow(2, self._private_key, self._prime)

    def generate_shared_key(self, other_public_key: bytes) -> bytes:
        remote_key = int.from_bytes(other_public_key, "big")
        self._shared_key = pow(remote_key, self._private_key, self._prime)
        return self.get_shared_key()

    def get_private_key(self) -> bytes:
        return self._to_bytes(self._private_key)

    def get_public_key(self) -> bytes:
        return self._to_bytes(self._public_key)

    def get_shared_key(self) -> bytes:
        return self._to_bytes(self._shared_key)

class KA: # s and c
    # gen (sk, pk) pairs using Diffie-Hellman
    @staticmethod
    def gen():
        dh = DH()
        sk, pk =  dh.get_private_key(), dh.get_public_key()
        return sk, pk

    # gen 256-bit shared key of two users using SHA-256 and (sk, pk) pair
    @staticmethod
    def agree(sk, pk):
        dh = DH()
        dh.set_private_key(sk)
        shared_key = dh.generate_shared_key(pk)
        h = SHA256.new()
        h.update(shared_key)
        key_256 = h.digest()
        return key_256

class SS: # s and b
    # return n shares of a secret with t as threshold
    @staticmethod
    def share(secret, t, n):
        secret_bytes = pickle.dumps(secret)
        secret_hex = secret_bytes.hex()
        shares = SecretSharer.split_secret(secret_hex, t, n)
        return shares

    # construct a secret, only works with >=t shares
    @staticmethod
    def recon(shares):
        secret_hex = SecretSharer.recover_secret(shares)
        secret_bytes = bytes.fromhex(secret_hex)
        secret = pickle.loads(secret_bytes)
        return secret

class AE: # e
    # use AES to encrypt the plaintext (256-bit int)
    # nonce is selected to be KA.agree(sk, pk)
    @staticmethod
    def encrypt(key, nonce, plaintext):
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        ciphertext = cipher.encrypt(plaintext)
        return ciphertext

    # use AES to decrypt the ciphertext
    @staticmethod
    def decrypt(key, nonce, ciphertext):
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext)
        return plaintext