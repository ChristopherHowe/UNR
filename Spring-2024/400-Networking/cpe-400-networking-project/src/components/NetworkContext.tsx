import { createContext, useState, useCallback } from 'react';
import { Router, Host } from '@/models';

interface NetworkContextProps {
  getHost: (mac: string) => Host | undefined;
  addHost: (host: Host) => void;
  editHost: (updatedHost: Host) => void;
  getRouter: (mac: string) => Router | undefined;
  addRouter: (router: Router) => void;
  editRouter: (updatedRouter: Router) => void;
  getDeviceType: (mac: string) => 'router' | 'host' | undefined;
}

export const NetworkContext = createContext<NetworkContextProps>({
  getHost: (mac: string) => undefined,
  addHost: (host: Host) => {},
  editHost: (updatedHost: Host) => {},
  getRouter: (mac: string) => undefined,
  addRouter: (router: Router) => {},
  editRouter: (updatedRouter: Router) => {},
  getDeviceType: (mac: string) => undefined,
});

export function NetworkContextProvider({ children }: { children: any }) {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [routers, setRouters] = useState<Router[]>([]);

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
    if (!getHost(updatedRouter.macAddress)) {
      throw new Error(`Router with mac address ${updatedRouter.macAddress} does not exist`);
    }
    setHosts((prev) => prev.map((router) => (router.macAddress !== updatedRouter.macAddress ? router : updatedRouter)));
  }

  function getDeviceType(mac: string): 'router' | 'host' | undefined {
    if (routers.some((router) => router.macAddress === mac)) {
      return 'router';
    }
    if (hosts.some((host) => host.macAddress === mac)) {
      return 'host';
    }
  }

  return (
    <NetworkContext.Provider value={{ getHost, addHost, editHost, getRouter, addRouter, editRouter, getDeviceType }}>
      {children}
    </NetworkContext.Provider>
  );
}
