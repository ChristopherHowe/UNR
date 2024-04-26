import SmoothDialog from './Dialog';
import { Fragment, useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { CheckIcon } from '@heroicons/react/24/outline';
import Textbox from './Textbox';

interface AddHostDialogProps {
  open: boolean;
  onClose: () => void;
  addHost: (name: string) => void;
}

export default function AddHostDialog(props: AddHostDialogProps) {
  const { open, onClose, addHost } = props;
  const [hostName, setHostName] = useState<string>('');

  function onSubmit() {
    addHost(hostName);
    onClose();
  }

  return (
    <SmoothDialog title="Add a New Host" {...{ open, onClose, onSubmit }}>
      <Textbox label="Host name" value={hostName} setValue={setHostName} />
    </SmoothDialog>
  );
}
