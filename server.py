from flask import *
from flask_socketio import SocketIO, emit
from flask_socketio import *
import json
import codecs
import pickle
import numpy as np


class secaggserver:
    def __init__(self, port, n, k):
        self.n = n  # the number of users
        self.k = k  # the threshold (t), should be separated to make things clearer
        self.aggregate = np.zeros((10, 10))
        self.port = port
        self.numkeys = 0 # number of public keys (S_u pk) received
        self.responses = 0 # number of encoded weights (X_i) received
        self.secretresp = 0 # number of PRG(b_u) received
        self.othersecretresp = 0 # number of offline users' secret key received
        self.respset = set()  # record the users that are `offline``
        self.resplist = []  # record the users that are `online`
        self.ready_client_ids = set()
        self.client_keys = dict()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()

    def weights_encoding(self, x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    def weights_decoding(self, s):
        return pickle.loads(codecs.decode(s.encode(), 'base64'))

    def register_handles(self):
        # when new client connects
        @self.socketio.on("connect")
        def handle_connect():
            self.ready_client_ids.add(request.sid)
            print(request.sid, " Connected")
            print('Connected devices:', self.ready_client_ids)

        # when the client emits `wakeup` signal
        @self.socketio.on("wakeup")
        def handle_wakeup():
            print("Recieved wakeup from", request.sid)
            emit("send_public_key", {
                "message": "hey I'm server",
                "id": request.sid
            })

        # when the server receives pubkey from client
        @self.socketio.on('public_key')
        def handle_pubkey(key):
            self.client_keys[request.sid] = key['key']
            self.numkeys += 1
            self.respset.add(request.sid)

            # current user's pubkey
            print(request.sid, 'sent key:', key['key'])
            print('keys: ', self.client_keys)  # all received pubkeys

            # if the all clients have sent the pubkeys, start broadcasting
            if (self.numkeys == self.n):
                print("Starting public key transfer")
                ready_clients = list(self.ready_client_ids)
                key_json = json.dumps(self.client_keys)
                for rid in ready_clients:
                    emit('share_weights', key_json, room=rid)

		# when the server receives weights (X_i)
        @self.socketio.on('weights')
        def handle_weights(data):
            print(request.sid, "sent weights")
            if self.responses < self.k:
				# decode the weights 
                self.aggregate += self.weights_decoding(data['weights'])
				# for online users, ask for PRG(b_u)
                emit('send_secret', {
                    'msg': "Hey I'm server"
                })
                print("MESSAGE SENT TO", request.sid)
                self.responses += 1
                self.respset.remove(request.sid)
                self.resplist.append(request.sid)
            else:
				# if the user arrives after k others, disconnects
				# (this is a naive approach that we can modify)
                emit('late', {
                    'msg': "Hey I'm server"
                })
                self.responses += 1
            if self.responses == self.k:
                print("k WIGHTS RECIEVED. BEGINNING AGGREGATION PROCESS.")
				# in this implementation, we are granted that we need to 
				# ask other online users for secret shares as we disconnet users k-t
                absentkeyjson = json.dumps(list(self.respset))
                for rid in self.resplist:
                    emit('reveal_secret', absentkeyjson, room=rid)

		# when the server receives individual masks (PRG(b_u))
        @self.socketio.on('secret')
        def handle_secret(data):
            print(request.sid, "sent SECRET")
            self.aggregate -= self.weights_decoding(data['secret']) # subtract to cancel out
            self.secretresp += 1
			
			# (similarly, we can separate k & t to be more flexible and make things clearer)
            if self.secretresp == self.k and self.othersecretresp == self.k:
                print("FINAL WEIGHTS:", self.aggregate)
                return self.aggregate

		# for each online user, asks PRG(S_u sk ** S_v pk), different from the paper
        @self.socketio.on('rvl_secret')
        def handle_secret_reveal(data):
            print(request.sid, "sent shared secrets")
            self.aggregate += self.weights_decoding(data['rvl_secret'])
            self.othersecretresp += 1
            if self.secretresp == self.k and self.othersecretresp == self.k:
                print("FINAL WEIGHTS:", self.aggregate)
                return self.aggregate

		# when someone disconnects, remove it from the ready_client_ids
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
            print(self.ready_client_ids)

    def start(self):
        self.socketio.run(self.app, port=self.port)


if __name__ == '__main__':
    server = secaggserver(2019, 3, 2)
    print("listening on http://localhost:2019")
    server.start()
