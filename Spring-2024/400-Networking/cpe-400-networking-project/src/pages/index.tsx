import Diagram from "@/components/Diagram";
import Layout from "@/components/Layout";
import { Controls } from "@/components/Controls";
import { useState } from 'react';
import  {Node,Edge, } from 'reactflow';
import SmoothDialog from "@/components/Dialog";
import AddHostDialog from "@/components/AddHostDialog";


interface Host {
  name: string
  macAddress: string
  ipAddress?: string
}
const usedMACAddresses: Set<string> = new Set();

function randomMacAddress(): string {
  return "XX:XX:XX:XX:XX:XX".replace(/X/g, function() {
    return "0123456789ABCDEF".charAt(Math.floor(Math.random() * 16))
  });
}

async function generateUniqueMACAddress(): Promise<string> {
  let macAddress: string;
  do {
    macAddress = randomMacAddress()
  } while (usedMACAddresses.has(macAddress)); 
  usedMACAddresses.add(macAddress);
  return macAddress;
}

export default function Home(){
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [hosts, setHosts] = useState<Host[]>([]);
  const [addHostDialopOpen, setaddHostDialopOpen] = useState<boolean>(false);

  async function addHost(name: string){
    const hostMACAddr = await generateUniqueMACAddress()
    console.log("adding a new host")
    setHosts(prevHosts => [...prevHosts, {
      name: name,
      macAddress: hostMACAddr,
    }])
  }

  function openAddHostDialog(){
    setaddHostDialopOpen(true)
  }

  function closeHostDialog(){
    setaddHostDialopOpen(false)
  }

  return(
    <>
      <Layout>
        <div className="relative h-full">
          <Diagram {...{nodes, edges}}/>
          <Controls addHost={openAddHostDialog}/>
        </div>
      </Layout>
      <AddHostDialog open={addHostDialopOpen} onClose={closeHostDialog} addHost={addHost}/>
    </>
    
  )
}
