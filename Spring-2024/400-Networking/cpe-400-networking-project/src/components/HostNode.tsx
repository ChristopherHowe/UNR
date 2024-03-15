import { Handle, NodeProps, Position } from 'reactflow';
import { Host } from "@/models";


interface NodeData {
    host: Host
}
export default function HostNode({ data }: NodeProps<NodeData>){
  return(
    <div className='border-2 bg-gray-200 border-black p-1 w-28 h-28 flex flex-col items-center justify-center rounded-full'>
      <div className='text-sm'>
        {data.host.name}
      </div>
      <div className='text-xs text-gray-400'>
        {data.host.macAddress}
      </div>
      {data.host.ipAddress && (
        <div className='text-xs text-gray-400'>
          {data.host.ipAddress}
      </div>
      )}
    </div>
  );
}
