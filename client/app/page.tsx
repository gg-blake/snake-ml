"use client"
//import Image from "next/image";
import { useEffect } from "react";

let keyPressed = "d";

export default function Home() {

  useEffect(() => {
    // Open a new WebSocket connection
    const control_channel = new WebSocket('ws://localhost:6600');
    // WebSocket open event
    control_channel.onopen = async function(event) {
        console.log("WebSocket control_channel opened:", event);
        control_channel.send(keyPressed);
        //control_channel.send("Pinging from the client on control_channel (Port:6600)");
    };

    // WebSocket message event
    control_channel.onmessage = async function() {  
      window.onkeydown = function(event) {
            if (event.key != keyPressed){
              keyPressed = event.key;
            }
        }
        control_channel.send(keyPressed);
    };
  }, []);

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      
    </div>
  );
}
