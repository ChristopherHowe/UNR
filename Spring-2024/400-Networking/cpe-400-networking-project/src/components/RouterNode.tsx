import { Handle, NodeProps, Position } from 'reactflow';
import { Router } from '@/models';

interface NodeData {
  router: Router;
  id: string;
}
export default function RouterNode({ data }: NodeProps<NodeData>) {
  return (
    <div className="host-node border-2 bg-gray-200 border-black p-1 w-28 h-28 flex flex-col items-center justify-center rounded-full">
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
      <Handle
        type="target"
        id={'bottom-target'}
        position={Position.Bottom}
        style={{ background: '#555' }}
        onConnect={(params) => console.log('handle target onConnect', params)}
        isConnectable
      />
      <Handle
        type="source"
        id={'right-source'}
        position={Position.Right}
        style={{ background: '#f00' }}
        onConnect={(params) => console.log('handle source onConnect', params)}
        isConnectable
      />
      <div className="text-sm">{data.router.name}</div>
      <div className="text-xs text-gray-400">{data.router.macAddress}</div>
      <div className="text-xs text-gray-400">{data.router.ipAddress}</div>
      <div className="text-xs text-gray-400">{data.router.subnet}</div>
    </div>
  );
}
