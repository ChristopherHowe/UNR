import SmoothDialog from './Dialog';
import { useState } from 'react';
import Textbox from './Textbox';
import { Router } from '@/models';
import { generateUniqueMACAddress } from '@/utils';

interface AddHostDialogProps {
  open: boolean;
  onClose: () => void;
  addRouter: (router: Router) => void;
}

export default function AddRouterDialog(props: AddHostDialogProps) {
  const { open, onClose, addRouter } = props;
  const [hostName, setHostName] = useState<string>('');
  const [ipAddr, setIPAddr] = useState<string>('');
  const [macAddr, setMacAddr] = useState<string>('');
  const [subnet, setSubnet] = useState<string>('');

  async function onSubmit() {
    const newMac = await generateUniqueMACAddress();
    addRouter({
      name: hostName,
      macAddress: macAddr !== '' ? macAddr : newMac,
      ipAddress: ipAddr,
      subnet: subnet,
    });
    onClose();
  }

  return (
    <SmoothDialog title="Add a New Host" {...{ open, onClose, onSubmit }}>
      <Textbox label="Host Name" value={hostName} setValue={setHostName} />
      <Textbox label="Ip Address" value={ipAddr} setValue={setIPAddr} />
      <Textbox label="Subnet" value={subnet} setValue={setSubnet} />
      <Textbox label="Mac Address (optional)" value={macAddr} setValue={setMacAddr} />
    </SmoothDialog>
  );
}
