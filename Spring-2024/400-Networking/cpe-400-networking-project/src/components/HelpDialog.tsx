import SmoothDialog from './Dialog';
import { useState } from 'react';
import Textbox from './Textbox';
import { Router } from '@/models';
import { generateUniqueMACAddress } from '@/utils';
import * as ip from 'ip';
import { isCIDRFormat } from '@/utils/network';

interface AddHostDialogProps {
  open: boolean;
  onClose: () => void;
}

export default function HelpDialog(props: AddHostDialogProps) {
  const { open, onClose } = props;
  const [hostName, setHostName] = useState<string>('');
  const [ipAddr, setIPAddr] = useState<string>('');
  const [macAddr, setMacAddr] = useState<string>('');
  const [subnet, setSubnet] = useState<string>('');

  return (
    <SmoothDialog title="Help" {...{ open, onClose }} onSubmit={onClose} validationMsg="" submitLabel="Done">
      <h1 className="text-xl mb-2 mt-3">Connecting components</h1>
      <div className="flex flex-row gap-2">
        <div className="bg-black rounded-full h-5 w-5 flex justify-center items-center">
          <div className="bg-red-500 border-white rounded-full border-2 h-4 w-4" />
        </div>
        <p>Signifies internal router network connection</p>
      </div>
      <div className="flex flex-row gap-2">
        <div className="bg-black rounded-full h-5 w-5 flex justify-center items-center">
          <div className="bg-gray-600 border-white rounded-full border-2 h-4 w-4" />
        </div>
        <p>Signifies external connection</p>
      </div>
    </SmoothDialog>
  );
}
