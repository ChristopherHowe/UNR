import { Cog6ToothIcon, HomeIcon, UsersIcon, WifiIcon } from '@heroicons/react/24/outline';
import React from 'react';
import { Simulation } from '@/models';
import data from '../../examples/example-1.json';

function classNames(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}

export default function Sidebar({ loadSimulation }: { loadSimulation: (sim: Simulation) => void }) {
  const navigation = [
    { name: 'Create a New Simulation', onClick: () => {}, icon: HomeIcon, current: true },
    {
      name: 'Example simulation',
      onClick: () => {
        loadSimulation(data);
      },
      icon: UsersIcon,
      current: false,
    },
  ];
  return (
    <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-gray-900 px-6 pb-4">
      <div className="flex h-16 shrink-0 items-center">
        <WifiIcon className="h-8 w-auto text-white" />
      </div>
      <nav className="flex flex-1 flex-col">
        <ul role="list" className="flex flex-1 flex-col gap-y-7">
          <li>
            <ul role="list" className="-mx-2 space-y-1">
              {navigation.map((item) => (
                <li key={item.name}>
                  <button
                    onClick={item.onClick}
                    className={classNames(
                      item.current ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800',
                      'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold',
                    )}
                  >
                    <item.icon className="h-6 w-6 shrink-0" aria-hidden="true" />
                    {item.name}
                  </button>
                </li>
              ))}
            </ul>
          </li>
          <li className="mt-auto">
            <a
              href="#"
              className="group -mx-2 flex gap-x-3 rounded-md p-2 text-sm font-semibold leading-6 text-gray-400 hover:bg-gray-800 hover:text-white"
            >
              <Cog6ToothIcon className="h-6 w-6 shrink-0" aria-hidden="true" />
              Settings
            </a>
          </li>
        </ul>
      </nav>
    </div>
  );
}
