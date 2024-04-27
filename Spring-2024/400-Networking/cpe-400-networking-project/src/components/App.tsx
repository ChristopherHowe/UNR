import Diagram from '@/components/Diagram';
import Layout from '@/components/Layout';
import { Controls } from '@/components/Controls';
import { useState, useEffect, useCallback, useContext } from 'react';
import { addEdge, useNodesState, useEdgesState, Connection } from 'reactflow';
import AddHostDialog from '@/components/AddHostDialog';
import { Host, Router } from '@/models';
import AddRouterDialog from './AddRouterDialog';
import { findUnusedIP } from '@/utils/network';
import { NetworkContext } from './NetworkContext';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [openDialog, setOpenDialog] = useState<'' | 'AddHost' | 'AddRouter'>('');

  const { addHost, addRouter } = useContext(NetworkContext);

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

  const onConnect = useCallback(
    (connection: Connection) => {
      const srcNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);
      if (!srcNode) {
        throw new Error(`Failed to find src node for id ${connection.source}`);
      }
      if (!targetNode) {
        throw new Error(`Failed to find target node for id ${connection.target}`);
      }
      if (targetNode.type !== 'routerNode' || !targetNode.data.router) {
        throw new Error('Target must be router');
      }
      const newIP = findUnusedIP(targetNode.data.subnet, targetNode.data.leases);
      setEdges((eds) => addEdge({ ...connection }, eds));
    },
    [setEdges, nodes],
  );

  return (
    <>
      <Layout>
        <div className="relative h-full">
          <Diagram {...{ nodes, onNodesChange, edges, onEdgesChange, onConnect }} />
          <Controls
            setOpenDialog={setOpenDialog}
            runSimulation={() => {
              console.log('Running Sim');
            }}
          />
        </div>
      </Layout>
      <AddHostDialog open={openDialog === 'AddHost'} onClose={closeDialog} addHost={handleAddHost} />
      <AddRouterDialog open={openDialog === 'AddRouter'} onClose={closeDialog} addRouter={handleAddRouter} />
    </>
  );
}
