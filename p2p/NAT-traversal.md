# Interaction between two peers with different NAT-powered networks

## Note

This tutorial is written for an outdated version of hivemind (`0.9.9.post1`, the latest version at the time of releasing the paper).
Since then we have significantly improved NAT traversal support and made some API changes. Please consider reading the **[up-to-date tutorial](https://github.com/yandex-research/DeDLOC/blob/hivemind-1.0.0/p2p/NAT-traversal.md)** instead.

## Intro

This brief tutorial covers how two peers from different LAN (with NAT) can connect with each other. Whole p2p network configured from scratch step by step.

Participants:
* Public accessible nodes `public-1`, `public-2`, `public-3`, `public-4`
* Private node `private-1` (NAT A)
* Private node `private-2` (NAT B)

First of all, you need to run installation using the same instructions as in swav or albert experiments. Make sure you use the correct version of hivemind:

```bash
pip install hivemind==0.9.9.post1
```

Run all code below using jupyter notebooks, `python -m asyncio`, `ipython` or copy-paste to .py file and run it (don't forget to initialize the asyncio loop and wait at the end, e.g. with asyncio.Event, to prevent premature termination of the program).

## Set up public nodes

### Start initial node `public-1`

```python
from hivemind.p2p.p2p_daemon import P2P
node = await P2P.create()
print(await node._client.identify())
```
Reference output:
```
(
    <libp2p.peer.id.ID (QmPublic1PeerID)>,
    (
        <Multiaddr /ip4/<public-1-ip>/tcp/<public-1-port>>,
        <Multiaddr /ip4/127.0.0.1/tcp/<public-1-port>>,
        <Multiaddr /ip4/<public-1-ip>/udp/<public-1-port>/quic>,
        <Multiaddr /ip4/127.0.0.1/udp/<public-1-port>/quic>)
    )
)
```

### Connect `public-2`, `public-3` and `public-4` to `public-1`
```python
from hivemind.p2p.p2p_daemon import P2P
nodes = ['/ip4/<public-1-ip>/tcp/<public-1-port>/p2p/QmPublic1PeerID']
node = await P2P.create(bootstrap=True, bootstrap_peers=nodes)
print(await node._client.identify())
```
Reference output:
```
(
    <libp2p.peer.id.ID (QmPublic2/3/4PeerID)>,
    (
        <Multiaddr /ip4/<public-2/3/4-ip>/tcp/<public-2/3/4-port>,
        <Multiaddr /ip4/127.0.0.1/tcp/<public-2/3/4-port>>,
        <Multiaddr /ip4/<public-2/3/4-ip>/udp/<public-2/3/4-port>/quic>,
        <Multiaddr /ip4/127.0.0.1/udp/<public-2/3/4-port>/quic>)
    )
)
```

## Run receiver node `private-1`

```python
from hivemind.p2p.p2p_daemon import P2P
nodes = [
    '/ip4/<public-1-ip>/udp/<public-1-port>/quic/p2p/QmPublic1PeerID',
    '/ip4/<public-2-ip>/udp/<public-2-port>/quic/p2p/QmPublic2PeerID',
    '/ip4/<public-3-ip>/udp/<public-3-port>/quic/p2p/QmPublic3PeerID',
    '/ip4/<public-4-ip>/udp/<public-4-port>/quic/p2p/QmPublic4PeerID',
]

node = await P2P.create(bootstrap=True, bootstrap_peers=nodes, dht_mode='dht')

def handler(a):
    return b'Hello, World!'

await node.add_stream_handler('handler_name', handler)

print(await node._client.identify())
```

Reference output:
```
(
    <libp2p.peer.id.ID (QmPrivate1PeerID)>,
    (
        <Multiaddr /ip4/192.168.1.100/tcp/53937>,
        <Multiaddr /ip4/127.0.0.1/tcp/53937>,
        <Multiaddr /ip4/192.168.1.100/udp/53937/quic>,
        <Multiaddr /ip4/127.0.0.1/udp/53937/quic>,
        <Multiaddr /ip4/<private-1-ip>/udp/<private-1-port>/quic>)
    )
)
```

`192.168.1.100` and `127.0.0.1` are LAN addresses.

`/ip4/<private-1-ip>/udp/<private-1-port>/quic>` - Hole Punched address


## Run sender node `private-2`

```python
from hivemind.p2p.p2p_daemon import P2P
nodes = [
    '/ip4/<public-1-ip>/udp/<public-1-port>/quic/p2p/QmPublic1PeerID',
    '/ip4/<public-2-ip>/udp/<public-2-port>/quic/p2p/QmPublic2PeerID',
    '/ip4/<public-3-ip>/udp/<public-3-port>/quic/p2p/QmPublic3PeerID',
    '/ip4/<public-4-ip>/udp/<public-4-port>/quic/p2p/QmPublic4PeerID',
]

node = await P2P.create(bootstrap=True, bootstrap_peers=nodes, dht_mode='dht')

result = await node.call_peer_handler('QmPrivate1PeerID', 'handler_name', b'')
print(result)
```

Reference output:
```
b'Hello, World'
```

Private node `private-2` called `private-1` handler function and received the answer.
