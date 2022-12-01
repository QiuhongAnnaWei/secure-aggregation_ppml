from socketIO_client import SocketIO, LoggingNamespace
from random import randrange
import numpy as np
from copy import deepcopy
import codecs
import pickle
import json



class SecAggregator:
	def __init__(self,common_base,common_mod,dimensions,weights):
		self.secretkey = randrange(common_mod)
		self.base = common_base
		self.mod = common_mod
		self.pubkey = (self.base**self.secretkey) % self.mod
		self.sndkey = randrange(common_mod)
		self.dim = dimensions
		self.weights = weights
		self.keys = {}
		self.id = ''
	def public_key(self):
		return self.pubkey
	def set_weights(self,wghts,dims):
		self.weights = wghts
		self.dim = dims
	def configure(self,base,mod):
		self.base = base
		self.mod = mod
		self.pubkey = (self.base**self.secretkey) % self.mod
	def generate_weights(self,seed):
		np.random.seed(seed)
		return np.float32(np.random.rand(self.dim[0],self.dim[1]))
	def prepare_weights(self,shared_keys,myid):
		self.keys = shared_keys
		self.id = myid
		wghts = deepcopy(self.weights)
		for sid in shared_keys:
			if sid>myid:
				print "1",myid,sid,(shared_keys[sid]**self.secretkey)%self.mod
				wghts+=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
			elif sid<myid:
				print "2",myid,sid,(shared_keys[sid]**self.secretkey)%self.mod
				wghts-=self.generate_weights((shared_keys[sid]**self.secretkey)%self.mod)
		wghts+=self.generate_weights(self.sndkey)
		return wghts
	def reveal(self, keylist):
		wghts = np.zeros(self.dim)
		for each in keylist:
			print each
			if each<self.id:
				wghts-=self.generate_weights((self.keys[each]**self.secretkey)%self.mod)
			elif each>self.id:
				wghts+=self.generate_weights((self.keys[each]**self.secretkey)%self.mod)
		return -1*wghts
	def private_secret(self):
		return self.generate_weights(self.sndkey)


class secaggclient:
	def __init__(self,serverhost,serverport):
		self.sio = SocketIO(serverhost,serverport,LoggingNamespace)
		self.aggregator = SecAggregator(3,100103,(10,10),np.float32(np.full((10,10),3,dtype=int)))
		self.id = ''
		self.keys = {}

	def start(self):
		self.register_handles()
		print("Starting")
		self.sio.emit("wakeup")
		self.sio.wait()

	def configure(self,b,m):
		self.aggregator.configure(b,m)

	def set_weights(self,wghts,dims):
		self.aggregator.set_weights(wghts,dims)

	def weights_encoding(self, x):
		return codecs.encode(pickle.dumps(x), 'base64').decode()

	def weights_decoding(self, s):
		return pickle.loads(codecs.decode(s.encode(),'base64'))

	def register_handles(self):
		def on_connect(*args):
			msg = args[0]
			self.sio.emit("connect")
			print("Connected and recieved this message",msg['message'])

		def on_send_pubkey(*args):
			msg = args[0]
			self.id = msg['id']
			pubkey = {
				'key': self.aggregator.public_key()
			}
			self.sio.emit('public_key',pubkey)

		def on_sharedkeys(*args):
			keydict = json.loads(args[0])
			self.keys = keydict
			print("KEYS RECIEVED: ",self.keys)
			weight = self.aggregator.prepare_weights(self.keys,self.id)
			weight = self.weights_encoding(weight)
			resp = {
				'weights':weight
			}
			self.sio.emit('weights',resp)

		def on_send_secret(*args):
			secret = self.weights_encoding(-1*self.aggregator.private_secret())
			resp = {
				'secret':secret
			}
			self.sio.emit('secret',resp)			

		def on_reveal_secret(*args):
			keylist = json.loads(args[0])
			resp = {
				'rvl_secret':self.weights_encoding(self.aggregator.reveal(keylist))
			}
			self.sio.emit('rvl_secret',resp)


		def on_disconnect(*args):
			self.sio.emit("disconnect")
			print("Disconnected")
		self.sio.on('connect', on_connect)
		self.sio.on('disconnect', on_disconnect)
		self.sio.on('send_public_key',on_send_pubkey)
		self.sio.on('public_keys',on_sharedkeys)
		self.sio.on('send_secret', on_send_secret)
		self.sio.on('late',on_disconnect)
		self.sio.on('send_there_secret',on_reveal_secret)

if __name__=="__main__":
	s = secaggclient("127.0.0.1",2019)
	s.set_weights(np.zeros((10,10)),(10,10))
	s.configure(2,100255)
	s.start()
	print("Ready")