import { Edge, Node } from 'reactflow';

export interface Host {
  name: string;
  macAddress: string;
  ipAddress?: string;
  queuedPackets: Packet[];
  recievedPackets: Packet[];
}

export interface Packet {
  srcIP: string;
  destIP: string;
  data: string;
}

export interface Router {
  name: string;
  macAddress: string;
  intIPAddress: string;
  extIPAddress: string;
  subnet: string;
  activeLeases: Lease[];
  queuedPackets: Packet[];
}

export interface Lease {
  macAddress: string;
  ipAddress: string;
}

export interface Simulation {
  nodes: Node[];
  edges: Edge[];
  Hosts: Host[];
  Routers: Router[];
}
