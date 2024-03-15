import { Handle, NodeProps, Position } from 'reactflow';

interface NodeData {
    name: string
}
export default function HostNode({ data }: NodeProps<NodeData>){
  return(
    <div className='border-2 bg-gray-200'>
      Host Node
    </div>
  );
}
