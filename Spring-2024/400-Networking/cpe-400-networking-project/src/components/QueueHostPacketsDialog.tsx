import SmoothDialog from './Dialog';
import { useContext, useState } from 'react';
import Textbox from './Textbox';
import { Packet } from '@/models';
import Button from './Button';
import { NetworkContext } from './NetworkContext';

function QueuedPacketsTable({
  packets,
  setPackets,
}: {
  packets: Packet[];
  setPackets: React.Dispatch<React.SetStateAction<Packet[]>>;
}) {
  function deletePacket(removing: Packet) {
    setPackets((prev) =>
      prev.filter((packet) => !(packet.data === removing.data && packet.destIP === removing.destIP)),
    );
  }
  return (
    <div className="mt-8 max-w-[35em] truncate">
      <div className="sm:flex sm:items-center">
        <div className="sm:flex-auto">
          <h1 className="text-base font-semibold leading-6 text-gray-900">Queued Packets</h1>
        </div>
      </div>
      <div className="mt-3 flow-root">
        <div className="-mx-4 -my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
          <div className="inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
            <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
              <table className="min-w-full divide-y divide-gray-300">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">
                      Destination
                    </th>
                    <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                      Data
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                  {packets.map((packet) => (
                    <tr key={packet.data + packet.destIP}>
                      <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                        {packet.destIP}
                      </td>
                      <td
                        className="whitespace-nowrap px-3 py-4 text-sm text-gray-500"
                        style={{ maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis' }}
                      >
                        {packet.data}
                      </td>
                      <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                        <button onClick={() => deletePacket(packet)} className="text-indigo-600 hover:text-indigo-900">
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface QueueHostPacketsDialogProps {
  open: boolean;
  onClose: () => void;
}

export default function QueueHostPacketsDialog(props: QueueHostPacketsDialogProps) {
  const { open, onClose } = props;
  const [newData, setNewData] = useState<string>('');
  const [newDestIP, setNewDestIP] = useState<string>('');
  const [packets, setPackets] = useState<Packet[]>([]);
  const validationMsg = '';

  const { editHost, getHost, editMac } = useContext(NetworkContext);

  function addPacket() {
    setPackets((prev) => [...prev, { destIP: newDestIP, data: newData }]);
  }

  async function onSubmit() {
    const host = getHost(editMac);
    if (!host) {
      throw new Error(`Failed to get host for mac ${host}`);
    }
    editHost({ ...host, packets: packets });
    onClose();
  }

  return (
    <SmoothDialog title="Queue Packets" {...{ open, onClose, onSubmit, validationMsg }}>
      <div className="flex flex-row items-end">
        <Textbox label="Packet Data" value={newData} setValue={setNewData} />
        <Textbox label="Destination IP" value={newDestIP} setValue={setNewDestIP} />
        <div>
          <Button label="add" onClick={addPacket} />
        </div>
      </div>
      <QueuedPacketsTable {...{ packets, setPackets }} />
    </SmoothDialog>
  );
}
