export interface Host {
  name: string;
  macAddress: string;
  ipAddress?: string;
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
