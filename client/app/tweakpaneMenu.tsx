import { Blade, BladeState } from '@tweakpane/core';
import { useRef, useEffect } from "react";


function TweakPaneMenu({ params, viewParams, setParams, setViewParams }: { params: Presets, viewParams: ViewPresets, setParams: React.Dispatch<React.SetStateAction<Presets>>, setViewParams: React.Dispatch<React.SetStateAction<ViewPresets>>}) {
    const paneRef = useRef<Pane | null>(null);
    const paneContainerRef = useRef<HTMLDivElement | null>(null);
  
  
    useEffect(() => {
      if (!paneRef.current && paneContainerRef.current) {
        paneRef.current = new Pane({ container: paneContainerRef.current });
      }
  
      if (paneRef.current) {
        // Clear and rebuild pane
        paneRef.current.children.forEach((child) => paneRef.current!.remove(child));
  
        const tab = paneRef.current.addTab({
          pages: [{ title: "Training" }, { title: "Visual" }]
        });
  
        tab.pages[0].addBinding(params, 'n_dims', {
          options: {
            "2D": 2,
            "3D": 3,
            "4D": 4
          }
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "n_dims": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'ttl', {
          step: 25,
          min: 100,
          max: 1000
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "ttl": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'width', {
          step: 1,
          min: 10,
          max: 100
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "width": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'turn_angle', {
          step: Math.PI / 16,
          min: Math.PI / 16,
          max: Math.PI / 2
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "turn_angle": e.value != Math.PI / 2 ? e.value : -1
          }))
        })
    
        tab.pages[0].addBinding(params, 'speed', {
          step: 0.01,
          min: 0.05,
          max: 2
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "speed": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'n_snakes', {
          step: 1,
          min: 1,
          max: 10000
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "n_snakes": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'sigma', {
          step: 0.00001,
          min: 0.00001,
          max: 100
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "sigma": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'alpha', {
          step: 0.00001,
          min: 0.00001,
          max: 100
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "alpha": e.value
          }))
        })
    
        tab.pages[0].addBinding(params, 'hidden_layer_size', {
          step: 1,
          min: 1,
          max: 100
        }).on("change", (e) => {
          if (Number.isNaN(e.value)) return;
          setParams(prevParams => ({
            ...prevParams,
            "hidden_layer_size": e.value
          }))
        })
    
        tab.pages[1].addBinding(viewParams, 'colorMode', {
          options: {
            "alive": "alive",
            "best": "best"
          }
        }).on("change", (e) => {
          setViewParams(prevParams => ({
            ...prevParams,
            "colorMode": e.value
          }))
        })
  
        tab.pages[1].addBinding(viewParams, 'viewBounds').on("change", (e) => {
          setViewParams(prevParams => ({
            ...prevParams,
            "viewBounds": !viewParams.viewBounds
          }))
        })
      }
    }, []);
  
    return <div className="w-auto absolute top left" ref={paneContainerRef}></div>;
}