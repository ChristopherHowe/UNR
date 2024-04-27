import { Handle, NodeProps, Position } from 'reactflow';
import { Host } from '@/models';
import { useContext, useState, useEffect } from 'react';
import { NetworkContext } from './NetworkContext';

interface NodeData {
  mac: string;
}
export default function HostNode({ data }: NodeProps<NodeData>) {
  const { getHost } = useContext(NetworkContext);
  const [host, setHost] = useState<Host>({ macAddress: '', name: '' });

  useEffect(() => {
    const newHost = getHost(data.mac);
    if (newHost) {
      setHost(newHost);
    }
  }, [getHost, data.mac]);

  return (
    <div className="border-2 bg-blue-500 border-black p-1 w-36 h-16 flex flex-col items-center justify-center rounded-md">
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
        <div className="text-sm text-white">{host.name}</div>
        <div className="text-xs text-gray-300">{host.macAddress}</div>
        {host.ipAddress && <div className="text-xs text-gray-300">{host.ipAddress}</div>}
      </>
    </div>
  );
}
