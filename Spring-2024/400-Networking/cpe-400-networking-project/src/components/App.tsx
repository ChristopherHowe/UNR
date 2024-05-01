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
import runSimulation, { SimError, SimState } from '@/simulation';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [openDialog, setOpenDialog] = useState<'' | 'AddHost' | 'AddRouter'>('');
  const [simState, setSimState] = useState<SimState>(SimState.Off);
  const [simMsg, setSimMsg] = useState<string>('');

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

    const newIP = findUnusedIP(router);

    if (memberType === 'host') {
      editHost({ ...(member as Host), ipAddress: newIP, gateway: router.intIPAddress });
    } else {
      editRouter({ ...(member as Router), extIPAddress: newIP, gateway: router.intIPAddress });
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

  function setEdgeActive(src: string, dest: string, active: boolean) {
    function makeActiveEdge(e: Edge): Edge {
      return { ...e, animated: true, style: { stroke: 'red' } };
    }
    function makeOfflineEdge(e: Edge): Edge {
      return { ...e, animated: false, style: { stroke: 'black' } };
    }
    if (
      !edges.some(
        (edge) => (edge.source === src && edge.target === dest) || (edge.source === dest && edge.target === src),
      )
    ) {
      throw new Error(`Failed to set edge with src ${src} and dst ${dest} as active`);
    }
    if (active) {
      setEdges((prev) =>
        prev.map((edge) =>
          (edge.source === src && edge.target === dest) || (edge.source === dest && edge.target === src)
            ? makeActiveEdge(edge)
            : makeOfflineEdge(edge),
        ),
      );
    } else {
      setEdges((prev) =>
        prev.map((edge) =>
          (edge.source === src && edge.target === dest) || (edge.source === dest && edge.target === src)
            ? makeOfflineEdge(edge)
            : edge,
        ),
      );
    }
  }

  return (
    <>
      <Layout loadSimulation={loadSimulation}>
        <div className="relative h-full">
          <Diagram {...{ nodes, onNodesChange, edges, onEdgesChange, onConnect }} />
          <Controls
            setOpenDialog={setOpenDialog}
            runSimulation={() =>
              runSimulation(routers, hosts, editHost, getHost, setEdgeActive, setSimState, setSimMsg)
            }
            saveSimulation={() => saveSimulation(nodes, edges)}
          />
          {simState !== SimState.Off && (
            <CurrentEvent
              heading={simState}
              message={simMsg}
              dismissable={simState === SimState.Complete}
              onDismiss={() => setSimState(SimState.Off)}
            />
          )}
        </div>
      </Layout>
      <AddHostDialog open={openDialog === 'AddHost'} onClose={closeDialog} addHost={handleAddHost} />
      <AddRouterDialog open={openDialog === 'AddRouter'} onClose={closeDialog} addRouter={handleAddRouter} />
      <QueueHostPacketsDialog open={editMac !== ''} onClose={closeDialog} />
    </>
  );
}
