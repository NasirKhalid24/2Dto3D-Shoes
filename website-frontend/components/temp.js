import React, { Suspense, useState, useCallback, useEffect } from 'react';

import ImageView from './ImageView';
import ModelView from './ModelView';
import Dropzone from 'react-dropzone'

import { useLoader } from "@react-three/fiber"; 
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'

import CircularProgress from "@mui/material/CircularProgress"
import {unzip} from 'unzipit';

import placeholder from './placeholder.json';
import styles from "../styles/MainView.module.css";

const toBase64 = file => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});

function GLTF( { scene_url } ){
    const { scene } = useLoader(GLTFLoader, scene_url);
    return <ModelView scene={ scene }/>
}



function MainView(){

    const [state, setState] = useState({
        scene_url: "p.gltf",
        image_url: placeholder.src,
        submitted: false,
        loading: false,
        files: {
            acceptedFiles: [],
            rejectedFiles: [],
        }
    })

    const POST =  useCallback( async (acceptedFiles, rejectedFiles) => {

        setState({
            ...state,
            loading: true
        })

        if(rejectedFiles.length > 0){
            alert("One file at a time!")
            return
        }
        if(acceptedFiles.length < 1){
            alert("No file found!")
            return
        }
    
        const b64v = await toBase64(acceptedFiles[0])
        setState(state => ({
            ...state,
            image_url: b64v,
            submitted: true,
            files: {
                acceptedFiles: acceptedFiles,
                rejectedFiles: rejectedFiles
            }
        }) )
    }, [state.files.acceptedFiles, state.files.rejectedFiles, setState])

    useEffect(() => {

        if(state.submitted == false){

            setState({
                ...state,
                loading: false
            })

            return
        }

        const formData = new FormData();

		formData.append('file', state.files.acceptedFiles[0]);

		const file = fetch(
			'http://104.131.101.22/',
			{
				method: 'POST',
				body: formData,
			}
        )
        .then((response) => response)
        .then((result) => {
            console.log(result)
            return result.blob()
        })
        .then((result) => {
            console.log(result)
            return unzip(result)
        })
        .then((result) => {
            for (const [name, entry] of Object.entries(result)) {
                console.log(name, entry);
            }
            setState({
                ...state,
                submitted: false,
                loading: false
            })
        })
        .catch((error) => {
            console.error('Error:', error);
        });

    }, [state.submitted])

    return (
        <div>

            <div className={styles.container}>
                <ImageView src={state.image_url}/>
                <Suspense fallback={  <CircularProgress /> }>
                    <GLTF scene_url={state.scene_url} />
                </Suspense>
            </div>

            { state.loading ? 
            
            <CircularProgress />

            :

            <Dropzone 
            accept={'image/jpeg, image/png'}
            maxFiles={1}
            onDrop={ (acceptedFiles, rejectedFiles) => POST(acceptedFiles, rejectedFiles)}>
                {({getRootProps, getInputProps}) => (
                    <section className={styles.dropzone}>
                    <div {...getRootProps()}>
                        <input {...getInputProps()} />
                        <p>Drag here or click to upload!</p>
                    </div>
                    </section>
                )}
            </Dropzone>
        
            }
            

        </div>
    )
}

export default MainView