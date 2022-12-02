# Secure Aggregation for Privacy Preserving Machine Learning

An implementation of Secure Aggregation algorithm based on ”[Practical Secure Aggregation for Privacy-Preserving Machine Learning
(Bonawitz et. al)](https://eprint.iacr.org/2017/281.pdf)“ in Python.

Adpated from https://github.com/ammartahir24/SecureAggregation.

Dependencies: Flask, socketio and socketIO_client

`pip install Flask`

`pip install flask_socketio`

# Usage:
## Client side:
### Init:

`c = secaggclient(port)`

### Give weights needed to be transmitted (originally set to zero)

`c.set_weights(nd_numpyarray,dimensions_of_array)`

### Set common base and mod

`c.configure(common_base, common_mod)`

### start client side:

`c.start()`

## Server side:
### init:

`s = secaggserver(port,n,k)`

where n is number of selected clients for the round and k is number of client responses required before aggregation process begins

### start server side:

`s.start()`
