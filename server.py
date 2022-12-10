from flask import *
from flask_socketio import SocketIO, emit
from flask_socketio import *
from utils import *
import pickle
import numpy as np
import random
import time

server_port = 2019
threshold = 2
dim = (10,10)


class secaggserver:
    def __init__(self, port, dim, n, t):
        self.port = port
        self.dim = dim
        self.n = n  # the number of users
        self.t = t  # the threshold

        self.U_0, self.U_1, self.U_2, self.U_3, self.U_4 = [], [], [], [], []
        self.ready_client_ids = set()
    
        self.c_pk_dict, self.s_pk_dict, self.e_uv_dict = {}, {}, {}
        self.sk_shares_dict, self.b_shares_dict = {}, {}

        self.started = False
        self.aggregated_value = np.zeros(dim)
        
        self.lasttime = 0
        self.curr_round = -1 

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()

    # use a seed to generate a random mask with same shape as the input
    def gen_mask(self, seed):
        np.random.seed(seed)
        return np.float64(np.random.rand(self.dim[0], self.dim[1]))

    def register_handles(self):
        # when new client connects
        @self.socketio.on("connect")
        def handle_connect():
            if not self.started:
                print(request.sid, " Connected")
            else:
                print("Protocol has already begun.")
                emit("invalid")

        # Setup
        @self.socketio.on("wakeup")
        def handle_wakeup():
            self.ready_client_ids.add(request.sid)
            if len(self.ready_client_ids) == self.n:
                print("All clients connected -- Starting Round 0.")
                
                self.started = True
                ready_clients = list(self.ready_client_ids)
                self.U_0 = ready_clients
                self.ready_client_ids.clear()

                self.lasttime = time.time()
                self.curr_round = 0

                for client_id in ready_clients:
                    emit("advertise_keys", {"id": client_id}, room=client_id)

        # Round 0 (AdvertiseKeys)
        @self.socketio.on('done_advertise_keys')
        def handle_advertise_keys(resp):
            if (self.curr_round != 0) or request.sid not in self.U_0:
                emit('invalid')
            else:
                # Part 1: adding entries for punctual users
                # actually in round, or this is the first time we pass the limit
                if(time.time()-self.lasttime > time_max): 
                    print("Arrived too late for Round 0: AvertiseKeys")
                    emit('invalid')
                else:
                    self.ready_client_ids.add(request.sid)
                    public_keys = pickle.loads(resp)
                    self.c_pk_dict[request.sid] = public_keys['c_u_pk']
                    self.s_pk_dict[request.sid] = public_keys['s_u_pk']
                
                # Part 2: either start next round, abort, or keep on waiting for next client (default fall through)
                # start next round
                if (len(self.ready_client_ids) == num_clients) or \
                   (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                    self.curr_round = 1  
                    print(f"\nCollected keys from {len(self.ready_client_ids)} clients -- Starting Round 1.")
                    
                    ready_clients = list(self.ready_client_ids)
                    self.U_1 = ready_clients
                    print("U_1: ", ready_clients)
                    self.ready_client_ids.clear()
                    self.lasttime = time.time()
                    for client_id in ready_clients:
                        emit('share_keys', (pickle.dumps(self.c_pk_dict),
                            pickle.dumps(self.s_pk_dict)), room=client_id)
                # abort
                if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
                    print("Disconnect all")
                    for client_id in self.ready_client_ids: emit("invalid", room=client_id)

        # Round 1 (ShareKeys)
        @self.socketio.on('done_share_keys')
        def handle_share_keys(resp):
            if (self.curr_round != 1) or request.sid not in self.U_1:
                emit('invalid')
            else:
                # Part 1: adding entries for punctual users
                # actually in round, or this is the first time we pass the limit
                if(time.time()-self.lasttime > time_max): 
                    print("Arrived too late for Round 1: ShareKeys")
                    emit('invalid')
                else:
                    self.ready_client_ids.add(request.sid)
                    e_uv_dict = pickle.loads(resp)
                    # for every user v, record dict[v][u] = e_{u,v}
                    for key, value in e_uv_dict.items():
                        if key not in self.e_uv_dict:
                            self.e_uv_dict[key] = {}
                        self.e_uv_dict[key][request.sid] = value
                
                # Part 2: either start next round, abort, or keep on waiting for next client (default fall through)
                # start next round
                if len(self.ready_client_ids)== len(self.U_1) or \
                   (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                    self.curr_round = 2 
                    print(f"\nCollected e_uv from {len(self.ready_client_ids)} clients -- Starting Round 2.")
                    
                    ready_clients = list(self.ready_client_ids)
                    self.U_2 = ready_clients
                    print("U_2: ", ready_clients)
                    self.ready_client_ids.clear()
                    self.lasttime = time.time()
                    for client_id in ready_clients:
                        emit('masked_input_collection', pickle.dumps(self.e_uv_dict[client_id]), room=client_id)
                # abort
                if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
                    print("Disconnect all")
                    for client_id in self.ready_client_ids: emit("invalid", room=client_id)

        # Round 2 (MaskedInputCollection)
        @self.socketio.on('done_masked_input_collection')
        def handle_masked_input_collection(resp):
            if (self.curr_round != 2) or request.sid not in self.U_2:
                emit('invalid')
            else:
                # Part 1: adding entries for punctual users
                # actually in round, or this is the first time we pass the limit
                if(time.time()-self.lasttime > time_max): 
                    print("Arrived too late for Round 2: MaskedInputCollection")
                    emit('invalid')
                else:
                    # decode the masked input
                    self.ready_client_ids.add(request.sid)
                    masked_input = pickle.loads(resp)
                    self.aggregated_value += masked_input
                
                # Part 2: either start next round, abort, or keep on waiting for next client (default fall through)
                # start next round
                if len(self.ready_client_ids) == len(self.U_2)  or \
                   (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                    self.curr_round = 3
                    print( f"\nCollected y_u from {len(self.ready_client_ids)} clients -- Starting Round 3.")
                    
                    ready_clients = list(self.ready_client_ids)
                    self.U_3 = ready_clients
                    print("U_3", ready_clients)
                    self.ready_client_ids.clear()
                    self.lasttime = time.time()
                    for client_id in ready_clients:
                        emit('unmasking', pickle.dumps(self.U_3), room=client_id)
                # abort
                if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
                    print("Disconnect all")
                    for client_id in self.ready_client_ids: emit("invalid", room=client_id)


        # Round 3 (Unmasking)
        @self.socketio.on('done_unmasking')
        def handle_unmasking(resp):
            if (self.curr_round != 3) or request.sid not in self.U_3:
                emit('invalid')
            else:
                # Part 1: adding entries for punctual users
                # actually in round, or this is the first time we pass the limit
                if(time.time()-self.lasttime > time_max): 
                    print("Arrived too late for Round 3: Unmasking")
                    emit('invalid')
                else:
                    self.ready_client_ids.add(request.sid)
                    data = pickle.loads(resp)
                    for id, share in data[0].items():
                        if id not in self.sk_shares_dict: self.sk_shares_dict[id] = []
                        self.sk_shares_dict[id].append(share) 
                    for id, share in data[1].items():
                        if id not in self.b_shares_dict: self.b_shares_dict[id] = []
                        self.b_shares_dict[id].append(share)
                    print(f"\n{request.sid} sent secret shares.")
                    

                # Part 2: either compute result, abort, or keep on waiting for next client (default fall through)
                # compute result
                if len(self.ready_client_ids) == len(self.U_3) or \
                   (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                    self.curr_round = 4 # even though no more rounds after it
                    print( f"\nCollected y_u from {len(self.ready_client_ids)} clients -- Starting Round 4 (final round).")

                    ready_clients = list(self.ready_client_ids)
                    self.U_4 = ready_clients
                    print("U_4", ready_clients)
                    mask = np.zeros(self.dim)

                    # reconstruct s_{u,v} for all u in U2\U3
                    dropped_out_users = [u for u in self.U_2 if u not in self.U_3]
                    for u in dropped_out_users:
                        s_u_sk = SS.recon(self.sk_shares_dict[u])
                        
                        for v in self.U_3:
                            sv_pk = self.s_pk_dict[v]
                            shared_key = KA.agree(s_u_sk, sv_pk)
                            random.seed(shared_key)
                            s_uv = random.randint(0, 2**32 - 1)
                            if v > u:
                                mask += self.gen_mask(s_uv)
                            elif v < u:
                                mask -= self.gen_mask(s_uv)
                
                    # reconstruct b_u for all u in U3
                    for u in self.U_3:
                        b_u = SS.recon(self.b_shares_dict[u])
                        mask -= self.gen_mask(b_u)
                    
                    # subtract the mask to get the final value
                    self.aggregated_value += mask
                    print(f"final aggregate:\n", self.aggregated_value)
                    exit()
                    # TODO: resetting for next round of the algorithm
                    # num_clients = len(self.U_4)

                # abort
                if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
                    print("Disconnect all")
                    for client_id in self.ready_client_ids: emit("invalid", room=client_id)
                    print("could not compute final aggregate")
                    exit()


        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)

    def start(self):
        self.socketio.run(self.app, port=self.port)


if __name__ == '__main__':
    num_clients = 3
    time_max = 10

    # t_i represents the simulated group size of U_i (3,3,2,2)
    # t_1, t_2, t_3, t_4 = 3, 3, 2, 2
    # assert num_clients >= t_1 >= t_2 >= t_3 >= t_4 >= threshold

    server = secaggserver(server_port, dim, num_clients, threshold)
    print("listening on http://localhost:2019")
    server.start()
