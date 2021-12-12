import React from "react"
import { Suspense, useState, useMemo, useRef, useEffect, useLayoutEffect } from "react"
import { Canvas, useLoader, useThree } from "@react-three/fiber"; 
import { OrbitControls } from "@react-three/drei";

import { CircularProgress, Container } from "@mui/material";

// import useDidMountEffect from '../hooks/useDidMountEffect';
import * as THREE from "three";
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'


function Box( {scene_} ) {

    const copiedScene = useMemo(() => scene_.clone(), [scene_]);
    const prim = useRef();

    return <primitive ref={prim} object={copiedScene} />;
}

const ModelView = (props) => {
  
  const [state_json_string, setJSON] = useState( props.json_string );
  const [state_scene, setScene] = useState( null );
  
  // UseEffect 1
  useEffect(() => {
    
    setJSON( props.json_string )
    setScene( null )

  }, [props.json_string])

  useEffect(() => {
    const loader = new GLTFLoader();
    const scene = new THREE.Scene();
    loader.parse(props.json_string, undefined, (onload) => { 
      console.log( onload.scene.children[0].geometry.attributes.position.array[0])
      scene.add( onload.scene ) 
      setScene( scene )
    }, (onerror) => console.log("Error Loading"));
  }, [state_json_string])

  return (
      <div>

        {
          state_scene === null ?

          <div style={{"width": "100%", "height": "100%"} }>
            <CircularProgress />
          </div>

          :

          <Canvas camera={{ position: [-0.0, 0.1, -0.8] }}>
            <Suspense fallback={ <CircularProgress /> }>
                <Box scene_={state_scene} />
                <OrbitControls />
            </Suspense>
            <ambientLight />
          </Canvas>

        }
          
      </div>      
  );
}

export default ModelView