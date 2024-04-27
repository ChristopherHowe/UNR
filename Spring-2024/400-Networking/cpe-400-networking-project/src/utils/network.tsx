import * as ip from 'ip';
import { Router } from '@/models';

/**
 * Gets the first available ip address not contained in the leases array
 * @param cidr
 * @param leases
 * @returns
 */
export function findUnusedIP(router: Router) {
  const subnet = ip.cidrSubnet(router.subnet);
  const hostBits = 32 - subnet.subnetMaskLength;
  const min = subnet.firstAddress;

  for (let i = 0; i < 2 ** hostBits; i++) {
    const currentIP = ip.fromLong(ip.toLong(min) + i);
    if (!subnet.contains(currentIP)) {
      throw new Error('Generated an ip not contained in the subnet');
    }
    if (!router.activeLeases.some((lease) => lease.ipAddress === currentIP)) {
      if (!(router.ipAddress === currentIP)) {
        return currentIP;
      }
    }
  }
  throw new Error(`Failed to find a valid IP for router ${router.macAddress}`);
}

export function isCIDRFormat(cidr: string): boolean {
  const parts = cidr.split('/');
  if (parts.length !== 2) return false; // CIDR should have two parts: IP address and subnet mask
  if (!ip.isV4Format(parts[0])) {
    return false;
  }
  // Validate subnet mask, should be a number between 0 and 32
  const subnetMask = parseInt(parts[1], 10);
  if (isNaN(subnetMask) || subnetMask < 0 || subnetMask > 32) {
    return false;
  }

  return true;
}
