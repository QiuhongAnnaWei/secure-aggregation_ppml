import socketio
import random
import numpy as np
import codecs
import pickle
from copy import deepcopy
from utils import *
import time

from server import server_port, threshold, dim


def sleep_for_a_while(s):
    print(f"### {s}: sleeping for 5 seconds")
    time.sleep(3)
    print(f"### {s}: woke up")

class SecAggregator:
    def __init__(self, t, dimensions, input):
        self.t = t
        self.dim = dimensions
        self.input = input
        self.keys = {}
        self.id = ''
        self.c_u_sk = None
        self.c_u_pk = None
        self.s_u_sk = None
        self.s_u_pk = None
        self.b_u = None
        self.b_u_share = None
        self.c_pk_dict = {}
        self.s_pk_dict = {}
        self.e_uv_dict = {}

    # set the `x` and dimension (shape)
    def set_input(self, input):
        self.input = input
        self.dim = self.input.shape

    # use a seed to generate a random mask with same shape as the input
    def gen_mask(self, seed):
        np.random.seed(seed)
        return np.float64(np.random.rand(self.dim[0], self.dim[1]))

    # handler for round 0
    def gen_keys(self):
        self.c_u_sk, self.c_u_pk = KA.gen()
        self.s_u_sk, self.s_u_pk = KA.gen()
        return self.c_u_pk, self.s_u_pk

    # handler for round 1
    def share_keys(self, U_1, c_pk_dict, s_pk_dict):
        # sample random b_u
        self.c_pk_dict = c_pk_dict
        self.s_pk_dict = s_pk_dict
        self.b_u = random.randint(0, 2**32 - 1)
        # generate secret shares of s_u_sk and b_u
        s_u_sk_shares = SS.share(self.s_u_sk, self.t, len(U_1))
        b_u_shares = SS.share(self.b_u, self.t, len(U_1))
        # for each other user v ∈ U1\{u}, compute e_{u,v}
        e_uv_dict = {}
        for i, v in enumerate(U_1):
            if v == self.id:
                self.b_u_share = b_u_shares[i]
                continue
            metadata = pickle.dumps(
                [self.id, v, s_u_sk_shares[i], b_u_shares[i]])
            c_v_pk = self.c_pk_dict[v]
            shared_key = KA.agree(self.c_u_sk, c_v_pk)
            e_uv = AE.encrypt(shared_key, shared_key, metadata)
            e_uv_dict[v] = e_uv
        return e_uv_dict

    # handler for round 2
    def prepare_masked_input(self, U_2, e_uv_dict):
        # u is not included in this `U_2` (this is really U_2\{u})
        masked_input = deepcopy(self.input)
        self.e_uv_dict = e_uv_dict
        for v in U_2:
            shared_key = KA.agree(self.s_u_sk, self.s_pk_dict[v])
            random.seed(shared_key)
            s_uv = random.randint(0, 2**32 - 1)
            if v > self.id:
                print("shared: (i < j)", self.id, v, s_uv)
                masked_input += self.gen_mask(s_uv)
            elif v < self.id:
                print("shared: (i > j)", self.id, v, s_uv)
                masked_input -= self.gen_mask(s_uv)
        masked_input += self.gen_mask(self.b_u)
        print("original input:\n", self.input)
        return masked_input

    # handler for round 3
    def unmasking(self, U_3):
        U_2 = list(self.e_uv_dict.keys())
        sk_shares_dict = {}
        b_shares_dict = {self.id: self.b_u_share}
        for v in U_2:
            c_v_pk = self.c_pk_dict[v]
            shared_key = KA.agree(self.c_u_sk, c_v_pk)
            metadata = pickle.loads(AE.decrypt(shared_key, shared_key, self.e_uv_dict[v]))
            assert metadata[0] == v and metadata[1] == self.id
            if v not in U_3:
                sk_shares_dict[v] = metadata[2] # for offline users, reconstruct s_v_sk
            else:
                b_shares_dict[v] = metadata[3] # for oneline users, reconstruct b_v
        return sk_shares_dict, b_shares_dict

class secaggclient:
    def __init__(self, serverport, t, dimensions, input):
        self.serverport = serverport
        self.aggregator = SecAggregator(t, dimensions, input)
        self.id = ''
        self.keys = {}
        self.sio = socketio.Client()
        self.register_handles()
        self.sio.connect("http://localhost:" + str(self.serverport))
        self.sio.wait()

    def configure(self, b, m):
        self.aggregator.configure(b, m)

    def set_input(self, input):
        self.aggregator.set_input(input)

    def input_encoding(self, x):
        return codecs.encode(pickle.dumps(x), 'base64')

    def register_handles(self):
        # Round 0 (AdvertiseKeys)
        @self.sio.on("advertise_keys")
        def on_advertise_keys(*args):
        
            msg = args[0]
            self.id = msg['id']
            sleep_for_a_while(f"CLIENT {self.id}: advertise_keys")

            self.aggregator.id = self.id
            c_u_pk, s_u_pk = self.aggregator.gen_keys()
            resp = {
                'c_u_pk': c_u_pk,
                's_u_pk': s_u_pk
            }
            self.sio.emit('done_advertise_keys', pickle.dumps(resp))

        # Round 1 (ShareKeys)
        @self.sio.on("share_keys")
        def on_share_keys(*args):
            sleep_for_a_while(f"CLIENT {self.id}: share_keys")

            c_pk_dict = pickle.loads(args[0])
            s_pk_dict = pickle.loads(args[1])
            U_1 = list(c_pk_dict.keys())
            print("U_1", U_1)
            e_uv_dict = self.aggregator.share_keys(U_1, c_pk_dict, s_pk_dict)
            self.sio.emit('done_share_keys', pickle.dumps(e_uv_dict))

        # Round 2 (MaskedInputCollection)
        @self.sio.on("masked_input_collection")
        def on_masked_input_collection(*args):
            sleep_for_a_while(f"CLIENT {self.id}: masked_input_collection")  

            e_uv_dict = pickle.loads(args[0])
            U_2 = list(e_uv_dict.keys())
            masked_input = self.aggregator.prepare_masked_input(
                U_2, e_uv_dict)
            self.sio.emit('done_masked_input_collection', pickle.dumps(masked_input))

        # Round 3 (Unmasking)
        @self.sio.on("unmasking")
        def on_unmasking(*args):
            sleep_for_a_while(f"CLIENT {self.id}: unmasking")

            U_3 = pickle.loads(args[0])
            sk_shares_dict, b_shares_dict = self.aggregator.unmasking(U_3)
            print(sk_shares_dict)
            print(b_shares_dict)
            self.sio.emit('done_unmasking', pickle.dumps([sk_shares_dict, b_shares_dict]))


        @self.sio.event
        def disconnect():
            self.sio.emit("disconnect")
            print("Disconnected!")
            self.sio.disconnect()


        @self.sio.on("waitandtry")
        def on_waitandtry():
            self.sio.sleep(1) # to make sure connection is fully set up
            print("\n\n\n!!!inside on_waitandtry")
            self.sio.emit("retryconnect")
                


if __name__ == "__main__":
    input = np.zeros(dim)
    c = secaggclient(server_port, threshold, dim, input)  # this input is a placeholder

    # here, you actually configure the input (must follow the same dim)
    # c.set_input(np.ones(dim))
    # c.set_input(np.random.rand(num_rows, num_cols))
