import Container from '@mui/material/Container'
import { width } from '@mui/system'
import Image from 'next/image'

const ImageView = (props) => {
    return (
        
        <img 
            src={props.src} alt="Shoe Image"
        />
    )
}

export default ImageView