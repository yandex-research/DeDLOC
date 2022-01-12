# Interaction between two peers from different networks behind NAT
## Intro

This brief tutorial covers how two peers from different LAN (with NAT) can communicate with each other. The whole p2p network is configured from scratch step-by-step.

Under the hood, hivemind uses [libp2p](https://libp2p.io/) and leverages its [NAT traversal features](https://docs.libp2p.io/concepts/nat/)
(e.g., [UDP hole punching](https://en.wikipedia.org/wiki/UDP_hole_punching)).

Participants:

* Public accessible node `public-1`
* (optional) Other public accessible nodes `public-2`, `public-3`, `public-4`
* Private node `private-1` (NAT A)
* Private node `private-2` (NAT B)

First of all, please install the actual version of hivemind:

```bash
pip install hivemind==1.0.0
```

Please run all code below using Jupyter notebooks, `python -m asyncio`, or `ipython`.

## Set up public nodes

### Start initial node `public-1`

```python
from hivemind.p2p import P2P

# By default, the libp2p daemon listens only to localhost for security reasons.
# We explicitly allow it to listen to all networks interfaces (at host 0.0.0.0) and
# both TCP and QUIC/UDP protocols.
node = await P2P.create(quic=True, host_maddrs=['/ip4/0.0.0.0/tcp/0', '/ip4/0.0.0.0/udp/0/quic'])

print('Peer ID:', node.peer_id)
print('Visible addresses:')
for addr in await node.get_visible_maddrs():
    print(addr)
```

Reference output:
```
Peer ID: QmPublic1PeerID
Visible addresses:
/ip4/<public-1-ip>/tcp/<public-1-port>/p2p/QmPublic1PeerID
/ip4/127.0.0.1/tcp/<public-1-port>/p2p/QmPublic1PeerID
/ip4/<public-1-ip>/udp/<public-1-port>/quic/p2p/QmPublic1PeerID
/ip4/127.0.0.1/udp/<public-1-port>/quic/p2p/QmPublic1PeerID
```

### (optional) Connect nodes `public-2`, `public-3` and `public-4` to node `public-1`

```python
from hivemind.p2p import P2P

# TODO: Insert a visible address of node public-1 here
initial_peers = ['/ip4/<public-1-ip>/tcp/<public-1-port>/p2p/QmPublic1PeerID']
node = await P2P.create(initial_peers=initial_peers,
                        quic=True, host_maddrs=['/ip4/0.0.0.0/tcp/0', '/ip4/0.0.0.0/udp/0/quic'])

print('Peer ID:', node.peer_id)
print('Visible addresses:')
for addr in await node.get_visible_maddrs():
    print(addr)
```

Reference output:
```
Peer ID: QmPublic2PeerID
Visible addresses:
/ip4/<public-2-ip>/tcp/<public-2-port>/p2p/QmPublic2PeerID
/ip4/127.0.0.1/tcp/<public-2-port>/p2p/QmPublic2PeerID
/ip4/<public-2-ip>/udp/<public-2-port>/quic/p2p/QmPublic2PeerID
/ip4/127.0.0.1/udp/<public-2-port>/quic/p2p/QmPublic2PeerID
```

## Run receiver node `private-1`

```python
from contextlib import closing
from hivemind.p2p import P2P

# TODO: Insert a visible address of node public-1 here
initial_peers = ['/ip4/<public-1-ip>/tcp/<public-1-port>/p2p/QmPublic1PeerID']
# You may add public visible addresses of other public nodes,
# so a node can connect even if node public-1 is down

node = await P2P.create(initial_peers=initial_peers,
                        quic=True, host_maddrs=['/ip4/0.0.0.0/tcp/0', '/ip4/0.0.0.0/udp/0/quic'],
                        dht_mode='auto')
# We set dht_mode='auto' so that the internal libp2p DHT can detect whether it's reachable from the outside

async def reverse_string(_, reader, writer):
    with closing(writer):
        s = await P2P.receive_raw_data(reader)
        # In Python, s[::-1] reverses a string
        await P2P.send_raw_data(s[::-1], writer)

await node.add_binary_stream_handler('reverse_string', reverse_string)

print('Peer ID:', node.peer_id)
print('Visible addresses:')
for addr in await node.get_visible_maddrs():
    print(addr)
```

Reference output:

```
Peer ID: QmPrivate1PeerID
Visible addresses:
/ip4/192.168.1.100/tcp/53937/p2p/QmPrivate1PeerID
/ip4/127.0.0.1/tcp/53937/p2p/QmPrivate1PeerID
/ip4/192.168.1.100/udp/53937/quic/p2p/QmPrivate1PeerID
/ip4/127.0.0.1/udp/53937/quic/p2p/QmPrivate1PeerID
/ip4/<private-1-ip>/udp/<private-1-port>/quic/p2p/QmPrivate1PeerID
```

`192.168.1.100` and `127.0.0.1` are LAN addresses.

`/ip4/<private-1-ip>/udp/<private-1-port>/quic` is a [hole-punched](https://en.wikipedia.org/wiki/UDP_hole_punching) address.

## Run sender node `private-2`

```python
from contextlib import closing
from hivemind.p2p import P2P, PeerID

# TODO: Insert a visible address of node public-1 here
initial_peers = ['/ip4/<public-1-ip>/tcp/<public-1-port>/p2p/QmPublic1PeerID']

node = await P2P.create(initial_peers=initial_peers,
                        quic=True, host_maddrs=['/ip4/0.0.0.0/tcp/0', '/ip4/0.0.0.0/udp/0/quic'],
                        dht_mode='auto')

# TODO: Insert a peer ID of node private-1 here
target = PeerID.from_base58('QmPrivate1PeerID')
_, reader, writer = await node.call_binary_stream_handler(target, 'reverse_string')
with closing(writer):
    await P2P.send_raw_data(b'Hello world!', writer)
    print(await P2P.receive_raw_data(reader))
```

Reference output:
```
b'!dlrow olleH'
```

Thus, private node `private-2` called `private-1`'s handler function and received the answer.
