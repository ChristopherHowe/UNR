import Diagram from '@/components/Diagram';
import Layout from '@/components/Layout';
import { Controls } from '@/components/Controls';
import { useState, useEffect, useCallback } from 'react';
import { Node, Edge, addEdge, useNodesState, useEdgesState } from 'reactflow';
import AddHostDialog from '@/components/AddHostDialog';
import { Host } from '@/models';

const usedMACAddresses: Set<string> = new Set();

function randomMacAddress(): string {
  return 'XX:XX:XX:XX:XX:XX'.replace(/X/g, function () {
    return '0123456789ABCDEF'.charAt(Math.floor(Math.random() * 16));
  });
}

async function generateUniqueMACAddress(): Promise<string> {
  let macAddress: string;
  do {
    macAddress = randomMacAddress();
  } while (usedMACAddresses.has(macAddress));
  usedMACAddresses.add(macAddress);
  return macAddress;
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [hosts, setHosts] = useState<Host[]>([]);
  const [addHostDialopOpen, setaddHostDialopOpen] = useState<boolean>(false);

  async function addHost(name: string) {
    const hostMACAddr = await generateUniqueMACAddress();
    console.log('adding a new host');
    setHosts((prevHosts) => [
      ...prevHosts,
      {
        name: name,
        macAddress: hostMACAddr,
        ipAddress: '129.12.12.123',
      },
    ]);
  }

  function openAddHostDialog() {
    setaddHostDialopOpen(true);
  }

  function closeHostDialog() {
    setaddHostDialopOpen(false);
  }

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

  const onConnect = useCallback(
    (params: any) => {
      console.log('Calling on connect with paras');
      console.log(params);
      setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#fff' } }, eds));
    },
    [edges, setEdges],
  );

  useEffect(() => {
    for (const host of hosts) {
      addHostNode(host);
    }
  }, [hosts]);

  return (
    <>
      <Layout>
        <div className="relative h-full">
          <Diagram {...{ nodes, onNodesChange, edges, onEdgesChange, onConnect }} />
          <Controls addHost={openAddHostDialog} />
        </div>
      </Layout>
      <AddHostDialog open={addHostDialopOpen} onClose={closeHostDialog} addHost={addHost} />
    </>
  );
}
