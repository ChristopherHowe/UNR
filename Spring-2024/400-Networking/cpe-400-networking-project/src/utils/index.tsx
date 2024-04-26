const usedMACAddresses: Set<string> = new Set();

function randomMacAddress(): string {
  return 'XX:XX:XX:XX:XX:XX'.replace(/X/g, function () {
    return '0123456789ABCDEF'.charAt(Math.floor(Math.random() * 16));
  });
}

export async function generateUniqueMACAddress(): Promise<string> {
  let macAddress: string;
  do {
    macAddress = randomMacAddress();
  } while (usedMACAddresses.has(macAddress));
  usedMACAddresses.add(macAddress);
  return macAddress;
}
