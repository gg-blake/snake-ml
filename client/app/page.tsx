"use client"
import Image from "next/image";
import StreamClient from "./stream";
import Pane from "tweakpane";
import { useEffect, useRef } from "react";

const TweakpaneComponent = ({ params, onParamsChange }: {params: any, onParamsChange: any}) => {
  const paneRef = useRef(null);

  useEffect(() => {
    if (paneRef.current === null) {
      const pane = new Pane();
      pane.on("change", (ev: any) => {
        onParamsChange(ev.value);
      });

      for (const [key, value] of Object.entries(params)) {
        pane.addInput(params, key);
      }

      if (paneRef.current) {
        paneRef.current = pane;
      }
      
    }
  }, [params, onParamsChange]);

  return <div ref={paneRef} />;
};


export default function Home() {
  

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <StreamClient />
      <TweakpaneComponent params={{"hello": 0}} onParamsChange={() => console.log("changed")} />
    </div>
  );
}
