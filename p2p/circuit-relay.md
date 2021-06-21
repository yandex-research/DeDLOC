# Relaying traffic between NAT peers via LibP2P Circuit Relay 

This tutorial covers how two peers from different LAN (with NAT) can communicate between eachother
via a relay peer.

Participants:
* Publically accessable relay node: `public-1`
* Private receiver node `private-1`
* Private sender node `private-2`

## Set up the relay node
```python
from hivemind.p2p.p2p_daemon import P2P

relay = await P2P.create(force_reachability='public', dht_mode='dht', auto_relay=True, relay_enabled=True)
print(await relay._client.identify())
```

Reference output:
```
Peer ID: QmPublic1PeerID
Peer Addrs:
/ip4/<public1-ip>/tcp/<port>
/ip4/127.0.0.1/tcp/<port>
/ip4/<public-ip>/udp/<port>/quic
/ip4/127.0.0.1/udp/<port>/quic
...
```

## Set up the receiver node
```python
from hivemind.p2p.p2p_daemon import P2P

bootstrap_node = '/ip4/<public1-ip>/udp/<port>/quic/p2p/<QmPublicPeerID>'
waiter = await P2P.create(force_reachability='private', dht_mode='dht',
                          auto_relay=True, relay_enabled=True,
                          bootstrap=True, bootstrap_peers=[bootstrap_node])
print(await waiter._client.identify())

def handler(a):
    return b'Hello, World!'

await waiter.add_stream_handler('handler_name', handler)
```

Reference output:
```
Peer ID: <QmPrivate1PeerID>
Peer Addrs:
/ip4/<private1-ip>/tcp/33945
/ip4/127.0.0.1/tcp/33945
/ip4/<private1-ip>/udp/33945/quic
/ip4/127.0.0.1/udp/33945/quic
...
```

## Set up the sender node
```python
from hivemind.p2p.p2p_daemon import P2P

bootstrap_node = '/ip4/<public1-ip>/udp/54557/quic/p2p/<QmPublic1PeerID>'
sender = await P2P.create(force_reachability='private', dht_mode='dht',
                          auto_relay=True, relay_enabled=True,
                          bootstrap=True, bootstrap_peers=[bootstrap_node])
print(await sender._client.identify())

result = await sender.call_peer_handler('<QmPrivate1PeerID>', 'handler_name', b'')
print(result)
```

Reference output:
```
Peer ID: <Private2PeerID>
Peer Addrs:
/ip4/<private2-ip>/tcp/<port>
/ip4/127.0.0.1/tcp/<port>
/ip4/<private2-ip>/udp/<port>/quic
/ip4/127.0.0.1/udp/<port>/quic
...
b'Hello, World!'
```
Private node `private-2` called `private-1` handler function and received the answer.