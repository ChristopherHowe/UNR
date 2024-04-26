import { Handle, NodeProps, Position } from 'reactflow';
import { Host } from '@/models';

interface NodeData {
  host: Host;
  id: string;
}
export default function HostNode({ data }: NodeProps<NodeData>) {
  id: return (
    <div className="border-2 bg-gray-200 border-black p-1 w-28 h-28 flex flex-col items-center justify-center rounded-full">
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
      <div className="text-sm">{data.host.name}</div>
      <div className="text-xs text-gray-400">{data.host.macAddress}</div>
      {data.host.ipAddress && <div className="text-xs text-gray-400">{data.host.ipAddress}</div>}
    </div>
  );
}
