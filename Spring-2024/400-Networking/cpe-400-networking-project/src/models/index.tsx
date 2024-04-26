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
}
