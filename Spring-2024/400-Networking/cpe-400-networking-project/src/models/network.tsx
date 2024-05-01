import { Edge, Node } from 'reactflow';

export interface Host {
  name: string;
  macAddress: string;
  gateway: string;
  ipAddress?: string;
  queuedPackets: Datagram[];
  recievedPackets: Datagram[];
}

export interface Datagram {
  srcIP: string;
  destIP: string;
  segment: Segment;
}

export interface Segment {
  srcPort: number;
  destPort: number;
  data: any;
}

export interface Router {
  name: string;
  macAddress: string;
  intIPAddress: string;
  extIPAddress: string;
  gateway: string;
  subnet: string;
  activeLeases: Lease[];
  queuedPackets: Datagram[];
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
