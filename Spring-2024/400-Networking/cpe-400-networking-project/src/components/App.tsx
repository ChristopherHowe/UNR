import Diagram from '@/components/Diagram';
import Layout from '@/components/Layout';
import { Controls } from '@/components/Controls';
import { useState, useEffect, useCallback, useContext } from 'react';
import { addEdge, useNodesState, useEdgesState, Connection, Edge, updateEdge } from 'reactflow';
import AddHostDialog from '@/components/AddHostDialog';
import { Host, Router, Simulation, Packet } from '@/models';
import AddRouterDialog from './AddRouterDialog';
import { findUnusedIP, getPath } from '@/utils/network';
import { NetworkContext } from './NetworkContext';
import QueueHostPacketsDialog from './HostPacketsDialog';
import sleep from '@/utils';
import CurrentEvent from './CurrentEvent';

enum SimState {
  Off,
  Running,
  SendingPacket,
  TranslatingAddress,
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [openDialog, setOpenDialog] = useState<'' | 'AddHost' | 'AddRouter'>('');
  const [simState, setSimState] = useState<SimState>(SimState.Off);

  const {
    hosts,
    routers,
    addHost,
    addRouter,
    getDeviceType,
    getHost,
    getRouter,
    editHost,
    editRouter,
    saveSimulation,
    editMac,
    setEditMac,
  } = useContext(NetworkContext);

  function addHostNode(mac: string) {
    setNodes((prevNodes) => [
      ...prevNodes,
      {
        id: mac,
        position: { x: 0, y: 0 },
        data: { mac: mac },
        type: 'hostNode',
      },
    ]);
  }

  function addRouterNode(mac: string) {
    setNodes((prevNodes) => [
      ...prevNodes,
      {
        id: mac,
        position: { x: 0, y: 0 },
        data: { mac: mac },
        type: 'routerNode',
      },
    ]);
  }

  async function handleAddHost(host: Host) {
    addHost(host);
    addHostNode(host.macAddress);
  }

  async function handleAddRouter(router: Router) {
    addRouter(router);
    addRouterNode(router.macAddress);
  }

  function closeDialog() {
    setOpenDialog('');
    setEditMac('');
  }

  function giveHostIP(connection: Connection) {
    const memberId = connection.source;
    const routerId = connection.target;
    if (!memberId || !routerId) {
      throw new Error('Connection source or target is null');
    }
    let member: Router | Host | undefined = getHost(memberId);
    let memberType: 'router' | 'host' = 'host';
    if (!member) {
      memberType = 'router';
      member = getRouter(memberId);
      if (!member) {
        throw new Error(`given member id ${memberId} doesn't correspond to a host or a router`);
      }
    }
    const router = getRouter(routerId);
    if (!router) {
      throw new Error(`given router id ${routerId} is invalid`);
    }
    console.log('Got member and router');
    console.log(member);
    console.log(router);

    const newIP = findUnusedIP(router);
    console.log(`got new IP ${newIP}`);

    if (memberType === 'host') {
      editHost({ ...(member as Host), ipAddress: newIP });
    } else {
      editRouter({ ...(member as Router), extIPAddress: newIP });
    }
    editRouter({ ...router, activeLeases: [...router.activeLeases, { ipAddress: newIP, macAddress: memberId }] });
  }

  const onConnect = useCallback(
    (connection: Connection) => {
      giveHostIP(connection);
      setEdges((eds) => addEdge({ ...connection }, eds));
    },
    [setEdges, nodes, giveHostIP],
  );

  function loadSimulation(sim: Simulation) {
    setNodes(sim.nodes);
    setEdges(sim.edges);
    for (const router of sim.Routers) {
      addRouter(router);
    }
    for (const host of sim.Hosts) {
      addHost(host);
    }
  }

  function makeActiveEdge(e: Edge): Edge {
    return { ...e, animated: true, style: { stroke: 'red' } };
  }
  function makeOfflineEdge(e: Edge): Edge {
    return { ...e, animated: false, style: { stroke: 'black' } };
  }

  async function translateNetwork() {}

  async function runSimulation() {
    async function showPath(path: string[]) {
      const stack = path.reverse();
      let current = stack.pop();
      while (stack.length >= 1) {
        let next = stack.pop();
        const newEdges = edges.map((edge) =>
          (edge.source === current && edge.target === next) || (edge.source === next && edge.target === current)
            ? makeActiveEdge(edge)
            : makeOfflineEdge(edge),
        );
        setEdges(newEdges);
        await sleep(1000);
        current = next;
      }
      setEdges((prev) => prev.map((edge) => makeOfflineEdge(edge)));
    }

    for (const host of hosts) {
      for (const packet of host.queuedPackets) {
        console.log('Simulating packet from ', host.macAddress);
        console.log(packet);
        const destHost = hosts.find((host) => host.ipAddress === packet.destIP); // WARNING THIS IS NOT IMPLEMENTING ROUTER PROTOCOL
        if (!destHost) {
          throw new Error('Failed to find destination host');
        }
        const path = getPath(host.macAddress, destHost.macAddress, edges);
        if (!path) {
          throw new Error(`Failed to get path from ${host.macAddress} to ${destHost.macAddress}`);
        }
        console.log('Path');
        console.log(path);
        await showPath(path);
        editHost({ ...host, queuedPackets: host.queuedPackets.filter((p) => p != packet) });
        console.log('Added packet to received packets for ${dest}');
        editHost({ ...destHost, recievedPackets: [...destHost.recievedPackets, packet] });
      }
    }
  }

  return (
    <>
      <Layout loadSimulation={loadSimulation}>
        <div className="relative h-full">
          <Diagram {...{ nodes, onNodesChange, edges, onEdgesChange, onConnect }} />
          <Controls
            setOpenDialog={setOpenDialog}
            runSimulation={runSimulation}
            saveSimulation={() => saveSimulation(nodes, edges)}
          />
          <CurrentEvent heading="Heading" message="Wow this is a really low message" />
        </div>
      </Layout>
      <AddHostDialog open={openDialog === 'AddHost'} onClose={closeDialog} addHost={handleAddHost} />
      <AddRouterDialog open={openDialog === 'AddRouter'} onClose={closeDialog} addRouter={handleAddRouter} />
      <QueueHostPacketsDialog open={editMac !== ''} onClose={closeDialog} />
    </>
  );
}
