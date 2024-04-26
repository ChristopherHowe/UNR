import Diagram from '@/components/Diagram';
import Layout from '@/components/Layout';
import { Controls } from '@/components/Controls';
import { useState, useEffect, useCallback } from 'react';
import { Node, Edge, addEdge, useNodesState, useEdgesState } from 'reactflow';
import AddHostDialog from '@/components/AddHostDialog';
import { Host, Router } from '@/models';
import AddRouterDialog from './AddRouterDialog';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [hosts, setHosts] = useState<Host[]>([]);
  const [routers, setRouters] = useState<Router[]>([]);

  const [openDialog, setOpenDialog] = useState<'' | 'AddHost' | 'AddRouter'>('');

  function addHostNode(host: Host) {
    setNodes((prevNodes) => [
      ...prevNodes,
      {
        id: host.macAddress,
        position: { x: 0, y: 0 },
        data: { host: host },
        type: 'hostNode',
      },
    ]);
  }

  function addRouterNode(router: Router) {
    setNodes((prevNodes) => [
      ...prevNodes,
      {
        id: router.macAddress,
        position: { x: 0, y: 0 },
        data: { router: router },
        type: 'routerNode',
      },
    ]);
  }

  async function addHost(host: Host) {
    setHosts((prevHosts) => [...prevHosts, host]);
    addHostNode(host);
  }

  async function addRouter(router: Router) {
    setHosts((prevRouters) => [...prevRouters, router]);
    addRouterNode(router);
  }

  function closeDialog() {
    setOpenDialog('');
  }

  const onConnect = useCallback(
    (params: any) => {
      console.log('Calling on connect with paras');
      console.log(params);
      setEdges((eds) => addEdge({ ...params }, eds));
    },
    [setEdges],
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
      <AddHostDialog open={openDialog === 'AddHost'} onClose={closeDialog} addHost={addHost} />
      <AddRouterDialog open={openDialog === 'AddRouter'} onClose={closeDialog} addRouter={addRouter} />
    </>
  );
}
