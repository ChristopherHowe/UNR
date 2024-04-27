import { Edge, Node } from 'reactflow';

export interface Host {
  name: string;
  macAddress: string;
  ipAddress?: string;
  packets: Packet[];
}

export interface Packet {
  destIP: string;
  data: string;
}

export interface Router {
  name: string;
  macAddress: string;
  ipAddress: string;
  subnet: string;
  activeLeases: Lease[];
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
