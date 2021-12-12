import React, { Suspense, useState, useCallback, useEffect } from 'react';

import ImageView from './ImageView';
import ModelView from './ModelView';
import Dropzone from 'react-dropzone'

import CircularProgress from "@mui/material/CircularProgress"

import placeholder from './placeholder.json';
import styles from "../styles/MainView.module.css";

const toBase64 = file => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});

function MainView(){

    const [state, setState] = useState({
        scene_url: JSON.stringify( placeholder.gltf ),
        image_url: placeholder.src,
        submitted: false,
        loading: false,
        files: {
            acceptedFiles: [],
            rejectedFiles: [],
        }
    })

    const POST =  useCallback( async (acceptedFiles, rejectedFiles) => {
        
        if(rejectedFiles.length > 0){
            alert("One file at a time!")
            setState({
                ...state,
                loading: false
            })
            return
        }
        if(acceptedFiles.length < 1){
            alert("No file found!")
            setState({
                ...state,
                loading: false
            })
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
    })

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
		formData.append('ext', state.files.acceptedFiles[0].type.split("/")[1]);

		const file = fetch(
			'/api/predict',
			{
				method: 'POST',
				body: formData,
			}
        )
        .then((response) => response.json())
        .then((result) => {
            
            setState({
                ...state,
                submitted: false,
                loading: false,
                files: {
                    acceptedFiles: [],
                    rejectedFiles: [],
                },
                scene_url: result["gltf"]
            })

        })
        .catch((error) => {
            console.error('Error:', error);
        });

    }, [state.image_url])

    return (
        <div>

            <div className={styles.container}>
                <ImageView src={state.image_url}/>
                <Suspense fallback={  <CircularProgress /> }>
                    <ModelView json_string={ state.scene_url } />
                </Suspense>
            </div>

            <Dropzone 
            accept={'image/jpeg, image/png, image/webp'}
            maxFiles={1}
            onDrop={ (acceptedFiles, rejectedFiles) => POST(acceptedFiles, rejectedFiles)}>
                {({getRootProps, getInputProps}) => (
                    <section className={styles.dropzone}>
                    <div {...getRootProps({className: state.loading ? 'dropzone disabled' : ''})} >
                        {/* {...getRootProps()} */}
                        <input {...getInputProps()} />
                        <p>Drag here or click to upload!</p>
                    </div>
                    </section>
                )}
            </Dropzone>

        </div>
    )
}

export default MainView