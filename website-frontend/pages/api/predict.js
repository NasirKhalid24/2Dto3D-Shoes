import os from "os";
import fs from "fs";
import { promises } from "fs";
import { mkdtemp } from 'fs/promises';
import path from "path";
import FormData from 'form-data';
import formidable from 'formidable';


import {unzip} from 'unzipit';
import obj2gltf from "obj2gltf"


export const config = {
    api: {
      bodyParser: false,
    },
}

export default async function handler(req, res){

    if (req.method == 'POST') {
    
        const form = new formidable.IncomingForm();
        form.keepExtensions = true;
        form.parse(req, async (err, fields, files) => {
            
            const f = files['file']
            const n = fields['ext']

            var formData = new FormData();
            formData.append('file', fs.createReadStream(f.filepath), `file.${n}`);

            const results = await fetch(
                'http://104.131.101.22/',
                {
                    method: 'POST',
                    body: formData,
                }
            )
            .then((response) => response)
            .then((result) => {
                return result.arrayBuffer()
            })
            .then((result) => {
                return unzip(result)
            })
            .then((result) => {
                return Object.entries(result['entries'])
            })
            .catch((error) => {
                console.error('Error:', error);
                return res.status(404).json({"helllo": "world"})
            });

            if(results === undefined){
                return res.status(404).json({"err": "Server offline for maintenance! Check back soon"})
            }
            const pth = path.join(os.tmpdir(), 'foo-')
            const filename = results[0][0].split(".")[0];
            const folder = await mkdtemp(pth, (err, folder) => {
                if (err) throw err;
            }); 

            await Promise.all(results.map(async ([name, entry]) => {
                    if(entry !== undefined){
                        const pp = path.join(folder, name)
                        const blob = await entry.arrayBuffer();
                        const data = await promises.writeFile(pp, Buffer.from( blob ))
                    }
            }))

            const gltf_model = await obj2gltf( path.join(folder, filename + ".obj") ).then(function (gltf) {
                return JSON.stringify(gltf);
            })

            await promises.unlink(f.filepath)
            await promises.rm(folder, {force: true, recursive: true});

            return res.status(200).json({"gltf": gltf_model})

        });
        
    } else {

        res.status(405).json({ error: `Method '${req.method}' Not Allowed` });

    }
    
}