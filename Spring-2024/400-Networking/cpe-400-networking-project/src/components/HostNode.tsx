import { Handle, NodeProps, Position } from 'reactflow';
import { Host } from '@/models';
import { useContext, useState, useEffect } from 'react';
import { NetworkContext } from './NetworkContext';

interface NodeData {
  mac: string;
}
export default function HostNode({ data }: NodeProps<NodeData>) {
  const { getHost, setEditMac } = useContext(NetworkContext);
  const [host, setHost] = useState<Host>({ macAddress: '', name: '', packets: [] });

  useEffect(() => {
    const newHost = getHost(data.mac);
    if (newHost) {
      setHost(newHost);
    }
  }, [getHost, data.mac]);

  return (
    <div className="border-2 bg-blue-500 border-black p-1 w-40 h-20 flex flex-col items-center justify-center rounded-md">
      <>
        <Handle
          type="target"
          id={'top-target'}
          position={Position.Top}
          style={{ background: '#555' }}
          onConnect={(params) => console.log('handle target onConnect', params)}
          isConnectable
        />
        <Handle
          type="source"
          id={'left-source'}
          position={Position.Left}
          style={{ background: '#f00' }}
          onConnect={(params) => console.log('handle source onConnect', params)}
          isConnectable
        />
        <div className="rounded-full bg-red-600 h-5 w-5"></div>
        <div className="text-sm text-white">{host.name}</div>
        <div className="text-xs text-gray-300">{host.macAddress}</div>
        {host.ipAddress && <div className="text-xs text-gray-300">{host.ipAddress}</div>}
        <button onClick={() => setEditMac(host.macAddress)} className="text-gray-300 hover:text-gray-50">
          Edit
        </button>
      </>
    </div>
  );
}
