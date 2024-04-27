import Diagram from '@/components/Diagram';
import Layout from '@/components/Layout';
import { Controls } from '@/components/Controls';
import { useState, useEffect, useCallback, useContext } from 'react';
import { addEdge, useNodesState, useEdgesState, Connection } from 'reactflow';
import AddHostDialog from '@/components/AddHostDialog';
import { Host, Router, Simulation } from '@/models';
import AddRouterDialog from './AddRouterDialog';
import { findUnusedIP } from '@/utils/network';
import { NetworkContext } from './NetworkContext';
import QueueHostPacketsDialog from './QueueHostPacketsDialog';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [openDialog, setOpenDialog] = useState<'' | 'AddHost' | 'AddRouter'>('');

  const { addHost, addRouter, getDeviceType, getHost, getRouter, editHost, editRouter, saveSimulation } =
    useContext(NetworkContext);

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
  }

  function giveHostIP(connection: Connection) {
    function giveIP(hostId: string, routerId: string) {
      const host = getHost(hostId);
      const router = getRouter(routerId);
      console.log('Got host and router');
      console.log(host);
      console.log(router);
      if (!host) {
        throw new Error(`given host id ${hostId} is invalid`);
      }
      if (!router) {
        throw new Error(`given router id ${routerId} is invalid`);
      }
      const newIP = findUnusedIP(router);
      console.log(`got new IP ${newIP}`);

      editHost({ ...host, ipAddress: newIP });
      editRouter({ ...router, activeLeases: [...router.activeLeases, { ipAddress: newIP, macAddress: hostId }] });
    }
    console.log(`connection`);
    console.log(connection);

    if (!connection.source || !connection.target) {
      throw new Error('Connection source or target is null');
    }
    const srcType = getDeviceType(connection.source);
    const targetType = getDeviceType(connection.target);
    console.log(`srcType: ${srcType}, targetType ${targetType}`);

    if (!srcType || !targetType) {
      throw new Error('Source or target type is undefined');
    }
    if (srcType === 'host' && targetType === 'host') {
      throw new Error('Connection source and router are both hosts');
    }
    if (srcType === 'router') {
      console.log(`giving ip from ${connection.source} to ${connection.target}`);
      giveIP(connection.target, connection.source);
    } else {
      console.log(`giving ip from ${connection.target} to ${connection.source}`);
      giveIP(connection.source, connection.target);
    }
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

  return (
    <>
      <Layout loadSimulation={loadSimulation}>
        <div className="relative h-full">
          <Diagram {...{ nodes, onNodesChange, edges, onEdgesChange, onConnect }} />
          <Controls
            setOpenDialog={setOpenDialog}
            runSimulation={() => {
              console.log('Running Sim');
            }}
            saveSimulation={() => saveSimulation(nodes, edges)}
          />
        </div>
      </Layout>
      <AddHostDialog open={openDialog === 'AddHost'} onClose={closeDialog} addHost={handleAddHost} />
      <AddRouterDialog open={openDialog === 'AddRouter'} onClose={closeDialog} addRouter={handleAddRouter} />
      <QueueHostPacketsDialog open={true} onClose={closeDialog} />
    </>
  );
}
