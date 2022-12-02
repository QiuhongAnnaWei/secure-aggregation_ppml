import socketio
from random import randrange
import numpy as np
from copy import deepcopy
import codecs
import pickle
import json


class SecAggregator:
    def __init__(self, common_base, common_mod, dimensions, weights):
        self.secretkey = randrange(common_mod)  # s_u secret
        self.pubkey = (self.base**self.secretkey) % self.mod  # s_u public
        self.sndkey = randrange(common_mod)  # individual mask (b)
        self.base = common_base
        self.mod = common_mod
        self.dim = dimensions
        self.weights = weights
        self.keys = {}
        self.id = ''

    # returns the public key
    def public_key(self):
        return self.pubkey

    # set the `x` and dimension (shape)
    def set_weights(self, wghts, dims):
        self.weights = wghts
        self.dim = dims

    # set the base and mod
    def configure(self, base, mod):
        self.base = base
        self.mod = mod
        self.pubkey = (self.base**self.secretkey) % self.mod

    # this is PRG(seed)
    def generate_weights(self, seed):
        np.random.seed(seed)
        return np.float32(np.random.rand(self.dim[0], self.dim[1]))

    # calculate x_u + sum_{u<v}PRG(s_{u,v}) - sum_{u>v}PRG(s_{v, u}) + PRG(b_u)
    # use the Diffie-Hellman key agreement to generate the public key
    def prepare_weights(self, shared_keys, myid):
        self.keys = shared_keys
        self.id = myid
        wghts = deepcopy(self.weights)
        for sid in shared_keys:
            if sid > self.id:
                # i < j case
                print("i < j", self.id, sid,
                      (shared_keys[sid]**self.secretkey) % self.mod)
                wghts += self.generate_weights(
                    (shared_keys[sid]**self.secretkey) % self.mod)
            elif sid < self.id:
                # i > j case
                print("i > j", self.id, sid,
                      (shared_keys[sid]**self.secretkey) % self.mod)
                wghts -= self.generate_weights(
                    (shared_keys[sid]**self.secretkey) % self.mod)
        wghts += self.generate_weights(self.sndkey)
        return wghts

    # reveal other's key (this is not using t-out-of-n shamir SS, we should implement that)
    def reveal(self, keylist):
        wghts = np.zeros(self.dim)
        for sid in keylist:
            if sid > self.id:
                wghts += self.generate_weights(
                    (self.keys[sid]**self.secretkey) % self.mod)
            elif sid < self.id:
                wghts -= self.generapyte_weights(
                    (self.keys[sid]**self.secretkey) % self.mod)
        return -1 * wghts

    def private_secret(self):
        # for online users, this is PRG(b_u)
        return self.generate_weights(self.sndkey)


class secaggclient:
    def __init__(self, serverport):
        self.aggregator = SecAggregator(
            3, 100103, (10, 10), np.float32(np.full((10, 10), 3, dtype=int)))
        self.id = ''
        self.keys = {}
        self.sio = socketio.Client()
        self.sio.connect("http://localhost:" + str(serverport))

    def start(self):
        self.register_handles()
        print("Starting")
        self.sio.emit("wakeup")  # triggers the server to ask for pubkey
        self.sio.wait()

    def configure(self, b, m):
        self.aggregator.configure(b, m)

    def set_weights(self, wghts, dims):
        self.aggregator.set_weights(wghts, dims)

    def weights_encoding(self, x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    def weights_decoding(self, s):
        return pickle.loads(codecs.decode(s.encode(), 'base64'))

    def register_handles(self):
        # standard event handlers
        @self.sio.event
        def connect(*args):
            msg = args[0]
            self.sio.emit("connect")
            print("Connected and recieved this message", msg['message'])

        @self.sio.event
        def connect_error(data):
            print("The connection failed!")
            self.sio.emit("disconnect")
            self.sio.disconnect()

        @self.sio.event
        def disconnect():
            self.sio.emit("disconnect")
            print("Disconnected!")

        # emit its pubkey to the server
        @self.sio.on("send_public_key")
        def on_send_pubkey(*args):
            msg = args[0]
            self.id = msg['id']
            pubkey = {
                'key': self.aggregator.public_key()
            }
            self.sio.emit('public_key', pubkey)

        # users reach n, now receiving all other's public keys and start sending weights
        @self.sio.on("share_weights")
        def on_sharedkeys(*args):
            keydict = json.loads(args[0])
            self.keys = keydict
            print("KEYS RECIEVED: ", self.keys)
            weight = self.aggregator.prepare_weights(self.keys, self.id)
            weight = self.weights_encoding(weight)
            resp = {
                'weights': weight
            }
            self.sio.emit('weights', resp)

        # if it arrives on-time, send RPG(b_u) to the server
        @self.sio.on("send_secret")
        def on_send_secret(*args):
            secret = self.weights_encoding(self.aggregator.private_secret())
            resp = {
                'secret': secret
            }
            self.sio.emit('secret', resp)

        # someone disconnected, so we need to reveal other's secret share
        @self.sio.on("reveal_secret")
        def on_reveal_secret(*args):
            keylist = json.loads(args[0])
            resp = {
                'rvl_secret': self.weights_encoding(self.aggregator.reveal(keylist))
            }
            self.sio.emit('rvl_secret', resp)

        # someone is late, disconnects
        @self.sio.on("late")
        def on_late(*args):
            self.sio.emit("disconnect")
            print("Arrived too late, disconnected!")


if __name__ == "__main__":
    c = secaggclient(2019)
    c.set_weights(np.zeros((10, 10)), (10, 10))
    c.configure(2, 100255)
    c.start()
    print("Ready")
