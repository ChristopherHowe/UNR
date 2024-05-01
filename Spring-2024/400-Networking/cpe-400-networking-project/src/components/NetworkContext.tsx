import { createContext, useState, useCallback, useEffect } from 'react';
import { Router, Host, Simulation } from '@/models/network';
import { Edge, Node } from 'reactflow';

interface NetworkContextProps {
  hosts: Host[];
  routers: Router[];
  getHost: (mac: string) => Host | undefined;
  addHost: (host: Host) => void;
  editHost: (updatedHost: Host) => void;
  getRouter: (mac: string) => Router | undefined;
  addRouter: (router: Router) => void;
  editRouter: (updatedRouter: Router) => void;
  getDeviceType: (mac: string) => 'router' | 'host' | undefined;
  saveSimulation: (nodes: Node[], edges: Edge[]) => void;
  editMac: string;
  setEditMac: (mac: string) => void;
}

export const NetworkContext = createContext<NetworkContextProps>({
  hosts: [],
  routers: [],
  getHost: (mac: string) => undefined,
  addHost: (host: Host) => {},
  editHost: (updatedHost: Host) => {},
  getRouter: (mac: string) => undefined,
  addRouter: (router: Router) => {},
  editRouter: (updatedRouter: Router) => {},
  getDeviceType: (mac: string) => undefined,
  saveSimulation: (nodes: Node[], edges: Edge[]) => '',
  editMac: '',
  setEditMac: (mac: string) => {},
});

export function NetworkContextProvider({ children }: { children: any }) {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [routers, setRouters] = useState<Router[]>([]);
  const [editMac, setEditMac] = useState<string>('');

  const getHost = useCallback(
    (mac: string) => {
      return hosts.find((host) => host.macAddress === mac);
    },
    [hosts],
  );

  function addHost(host: Host) {
    setHosts((prev) => [...prev, host]);
  }

  function editHost(updatedHost: Host) {
    if (!getHost(updatedHost.macAddress)) {
      throw new Error(`Host with mac address ${updatedHost.macAddress} does not exist`);
    }
    setHosts((prev) => prev.map((host) => (host.macAddress !== updatedHost.macAddress ? host : updatedHost)));
  }

  const getRouter = useCallback(
    (mac: string) => {
      return routers.find((host) => host.macAddress === mac);
    },
    [routers],
  );

  function addRouter(router: Router) {
    setRouters((prev) => [...prev, router]);
  }

  function editRouter(updatedRouter: Router) {
    if (!getRouter(updatedRouter.macAddress)) {
      throw new Error(`Router with mac address ${updatedRouter.macAddress} does not exist`);
    }
    setRouters((prev) =>
      prev.map((router) => (router.macAddress !== updatedRouter.macAddress ? router : updatedRouter)),
    );
  }

  function getDeviceType(mac: string): 'router' | 'host' | undefined {
    if (routers.some((router) => router.macAddress === mac)) {
      return 'router';
    }
    if (hosts.some((host) => host.macAddress === mac)) {
      return 'host';
    }
  }

  function saveSimulation(nodes: Node[], edges: Edge[]) {
    const simulation: Simulation = {
      nodes: nodes,
      edges: edges,
      Hosts: hosts,
      Routers: routers,
    };
    const blob = new Blob([JSON.stringify(simulation)], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'simulation.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <NetworkContext.Provider
      value={{
        hosts,
        routers,
        getHost,
        addHost,
        editHost,
        getRouter,
        addRouter,
        editRouter,
        getDeviceType,
        saveSimulation,
        editMac,
        setEditMac,
      }}
    >
      {children}
    </NetworkContext.Provider>
  );
}
