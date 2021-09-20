'use strict';
const cors = require('cors');
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
//global.cv = require('./opencv.js');

//cv['onRuntimeInitialized'] =()=>{
//console.log("opencv loaded")
//const {kycimage} = require('./kyc.js')
//let dst=new cv.Mat();
const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json({ limit: "50mb" }));

app.get('/api/model_info', (req, res) => {
  const date = fs.readFileSync('model_info.txt', 'utf8');
  return res.status(200).json({
    last_updated: date.trim()
  });
});

app.use(express.static(path.join(__dirname, 'build')));

app.get('/*', function (req, res) {
   res.sendFile(path.join(__dirname, 'build', 'index.html'));
});
/*
app.post('/api/kyc', async (req, res) => {
    //const req_body = JSON.parse(req.body);
    //console.log(req.body.imagebuf);
    const imageBuffer = req.body.imagebuf
    const height = parseInt(req.body.height)
    const width = parseInt(req.body.width)
    const channels = parseInt(req.body.channels)
    // Output the book to the console for debugging
    //console.log(req.body);
    //console.log(typeof req);
    //console.log(req);
    //console.log(imageBuffer);
    let probabilities = await kycimage(imageBuffer,height,width)
    res.send({'probabilities': probabilities})
})
*/
    //res.send(`Type of document: ${doctype}`);
app.listen(process.env.PORT || port);
console.log(`Running on http://localhost:${port}`);
//console.log(cv.getBuildInformation())
//}
