import { Router, Host, Datagram } from './models/network';
import sleep from './utils';

const speed = 2000;

export class SimError extends Error {
  constructor(message: string) {
    super(message);
    Object.setPrototypeOf(this, SimError.prototype);
  }
}

export enum SimState {
  Off = 'Sim is not running',
  Running = 'Running Simulation...',
  SendingPacket = 'Sendng packet...',
  FinishedPacket = 'Successfully sent a packet',
  TranslatingAddress = 'Handling address translation...',
  Error = 'Something went wrong',
  Complete = 'Finished Simulation',
}

function getDestIPMacOnNet(router: Router, destIp: string): string | undefined {
  return router.activeLeases.find((lease) => lease.ipAddress === destIp)?.macAddress;
}

function dropPacketFromQueueWhenDone(h: Host, completedPacket: Datagram, editHost: (updatedHost: Host) => void) {
  editHost({ ...h, queuedPackets: h.queuedPackets.filter((p) => p !== completedPacket) });
}

function receivePacket(h: Host, editHost: (updatedHost: Host) => void, p: Datagram) {
  editHost({ ...h, recievedPackets: [...h.recievedPackets, p] });
}

function getHostRouter(hostMac: string, hostGateway: string, routers: Router[]) {
  if (hostGateway != '') {
    return routers.find(
      (router) =>
        router.activeLeases.some((lease) => lease.macAddress === hostMac) && router.intIPAddress === hostGateway,
    );
  }
}

async function simulatePacket(
  packet: Datagram,
  routers: Router[],
  srcHost: Host,
  editHost: (updatedHost: Host) => void,
  getHost: (mac: string) => Host | undefined,
  setEdgeActive: (src: string, dest: string, active: boolean) => void,
  setSimState: React.Dispatch<React.SetStateAction<SimState>>,
  setSimMsg: React.Dispatch<React.SetStateAction<string>>,
) {
  console.log('Simulating Packet', JSON.stringify(packet));
  let current: { macAddress: string; gateway: string } = srcHost;
  while (true) {
    const hostRouter = getHostRouter(current.macAddress, current.gateway, routers);
    if (!hostRouter) {
      throw new SimError(`Host/Router with mac ${current.macAddress} is not connected to a router`);
    }
    console.log(`activating edge from ${current.macAddress} to ${hostRouter.macAddress}`);
    setEdgeActive(current.macAddress, hostRouter.macAddress, true);
    await sleep(speed);

    const destMac = getDestIPMacOnNet(hostRouter, packet.destIP);
    // Case 1: Destination is in network
    if (destMac) {
      console.log('handling case 1');
      const destHost = getHost(destMac);
      if (!destHost) {
        throw new Error(`Got destination mac addr ${destMac} but no corresponding host`);
      }

      console.log(`activating edge from ${hostRouter.macAddress} to ${destMac}`);
      setEdgeActive(hostRouter.macAddress, destMac, true);
      await sleep(speed);
      console.log(`disabling edge from ${hostRouter.macAddress} to ${destMac}`);

      setEdgeActive(hostRouter.macAddress, destMac, false);

      receivePacket(destHost, editHost, packet);
      return;
    }
    // Case 2: Destination is not in network
    else {
      console.log('Handling case 2');
      console.log('host router ', JSON.stringify(hostRouter));
      if (hostRouter.gateway !== '') {
        setEdgeActive(current.macAddress, hostRouter.macAddress, true);
        sleep(speed);
        current = hostRouter;
      } else {
        throw new SimError(
          `Failed to deliver packet, after reaching router with mac ${hostRouter.macAddress}, could not proceed to anymore routers`,
        );
      }
    }
  }
}

export default async function runSimulation(
  routers: Router[],
  hosts: Host[],
  editHost: (updatedHost: Host) => void,
  getHost: (mac: string) => Host | undefined,
  setEdgeActive: (src: string, dest: string, active: boolean) => void,
  setSimState: React.Dispatch<React.SetStateAction<SimState>>,
  setSimMsg: React.Dispatch<React.SetStateAction<string>>,
) {
  setSimState(SimState.Running);
  try {
    for (const host of hosts) {
      for (const packet of host.queuedPackets) {
        setSimState(SimState.SendingPacket);
        setSimMsg(`Handling new packet with destination ${packet.destIP} and source ${packet.srcIP}`);
        await sleep(speed);

        await simulatePacket(packet, routers, host, editHost, getHost, setEdgeActive, setSimState, setSimMsg);
        dropPacketFromQueueWhenDone(host, packet, editHost);

        setSimState(SimState.FinishedPacket);
        setSimMsg('Finished sending packet');
        await sleep(speed);
      }
    }
    setSimState(SimState.Complete);
    setSimMsg('Finished');
  } catch (e) {
    setSimState(SimState.Error);
    if (e instanceof SimError) {
      setSimMsg(e.message);
    } else {
      setSimMsg('An unexpected error occured while running the simulator');
      console.log(e);
    }
  }
}
