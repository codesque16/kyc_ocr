//import cv from "@techstark/opencv-js";
//import { createWorker } from 'tesseract.js';
//import PSM from 'tesseract.js/src/constants/PSM.js'
//import OEM from 'tesseract.js/src/constants/OEM.js'
//import path from 'path';
import * as tf from '@tensorflow/tfjs';
const { Image } = require('image-js');
//const IJS = require("vshushkov-imagejs");
//const baseblob = require('based-blob');

//const worker = createWorker({
//  langPath: path.join(__dirname, 'public', 'tesseract_data'),
//  logger: m => console.log(m),
//});
const MODEL_INPUT_SIZE = 200;
function preprocessImage(img,w,h) {
    let img_w=img.cols
    let img_h=img.rows
    let new_w = parseInt(img_w * Math.min(w/img_w, h/img_h))
    let new_h = parseInt(img_h * Math.min(w/img_w, h/img_h))
    //console.log(new_w,new_h)

    let dst = new window.cv.Mat();
    let dsize = new window.cv.Size(new_w,new_h);
    //console.log(dsize)
    //console.log(img_w,img_h,img.data)
    // You can try more different parameters
    window.cv.resize(img, dst, dsize, 0, 0, window.cv.INTER_CUBIC);
    let canvas = window.cv.Mat.zeros(h,w, window.cv.CV_8U);
    let col_start=~~((w-new_w)/2)
    let row_start=~~((h-new_h)/2)
    //console.log(row_start,col_start)
    for (let r = 0; r < new_h; r++){
      for(let c = 0; c < new_w; c++){
        canvas.ucharPtr(row_start+r,col_start+c)[0]=dst.ucharPtr(r,c)[0]
      }
    }

    return canvas
}

async function classifyDocument(preprocessed_img,model) {
  //const classes=['AADHAAR_EXTRACTION', 'DL_EXTRACTION', 'PAN_EXTRACTION',
  //'PASSPORT_EXTRACTION', 'VOTER_EXTRACTION']
  //var predicted="NO_OUTPUT";
  //let model = await tf.loadLayersModel('file://public/kyc_model/model.json');
  console.log(await tf.tensor(preprocessed_img.data).div(tf.scalar(255.0)).array())
  let scores = await model.predict([tf.tensor(preprocessed_img.data).div(tf.scalar(255.0)).reshape([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1])]);
  //console.log(await scores.data())
  let score_array= await scores.array()
  //let predicted=score_array[0].indexOf(Math.max(...score_array));
  return tf.softmax(score_array[0]).array()
}

/*async function getTextFromImage(buffer) {
  await worker.load()
  await worker.loadLanguage('eng')
  await worker.initialize('eng')
  await worker.setParameters({
    tessedit_pageseg_mode: PSM.AUTO,
    //tessedit_char_whitelist: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
    //preserve_interword_spaces: "1",
    //tessjs_create_box: "1",
    //tessedit_ocr_engine_mode: OEM.LSTM_ONLY
  })
  const {data} = await worker.recognize(buffer);
  await worker.terminate()

  //printKeys(data)
  //console.log(typeof data)
  return data['text']
}
*/
function getBlackHatImage(imageMat) {
  console.log(imageMat)
  window.cv.cvtColor(imageMat, imageMat, window.cv.COLOR_RGBA2GRAY);
  console.log(imageMat)
  let dst = new window.cv.Mat();
  console.log(dst)
  let M = window.cv.Mat.ones(25,7, window.cv.CV_8U);
  console.log(M)
  window.cv.morphologyEx(imageMat, dst, window.cv.MORPH_BLACKHAT, M);
  window.cv.cvtColor(dst, dst, window.cv.COLOR_GRAY2RGBA);
  console.log(dst)

  /*new Jimp({ data: dst.data, width: dst.cols, height: dst.rows }, (err, image) => {
  if (err) throw err;
  image.write('tesseract.png');
  // this image is 1280 x 768, pixels are loaded from the given buffer.
  });
  */
  return dst
}


async function getImage(buffer) {
    //return buffer
    let img = await Image.load(buffer);
//  let output = tf.browser.fromPixels(img);
    return img;
    // read from a file

// read JPG data from stream
    //let bitmap = new IJS.Bitmap();
    //console.log(baseblob.toBlob(buffer.split(',')[1]))
    //bitmap.read(baseblob.toBlob(buffer.split(',')[1]), { type: IJS.ImageType.PNG })
    //.then(function() {
    //    console.log('read');
    //});
    //return bitmap
    //let img = new Image();
    //img.onload = function() {
    //    console.log("Image created");
    //    return img;
    //};
    //img.src = buffer;
    //await img.decode();
    //console.log(img)
    //return img
}

const kycimage = async function(buffer,model) {
    //console.log(jsimg);
    //let src = buffer;
    let jsimg = await getImage(buffer);
    const src = await new window.cv.matFromImageData(jsimg)
    //src.convertTo(src, window.cv.CV_8U);
    console.log(src.rows)
    //console.log(src.cols)
    //console.log(src.data)
    /*new Jimp({ data: src.data, width: src.cols, height: src.rows }, (err, image) => {
          if (err) throw err;
          image.write('kyc_step1.png');
    });*/
    let dst = getBlackHatImage(src);
    console.log(dst);
    //new Jimp({ data: dst.data, width: dst.cols, height: dst.rows }, (err, image) => {
    //    if (err) throw err;
    //    image.write('tesseract.png');
    //});
    //let tesbase64 = 'data:image/png;base64,' + Buffer.from(dst.data,'binary').toString('base64');
    //console.log(await im2base64(Buffer.from(src.data)));
    //console.log(encode(src.data,[src.cols,src.rows],'png'))
    //await getTextFromImage('tesseract.png')
    //    .then(console.log);

    window.cv.cvtColor(dst, dst, window.cv.COLOR_RGBA2GRAY);
    console.log(dst);
    let pImage= preprocessImage(dst,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE);
    console.log(pImage);
    window.cv.cvtColor(pImage, pImage, window.cv.COLOR_GRAY2RGBA);
    /*new Jimp({ data: pImage.data, width: pImage.cols, height: pImage.rows }, (err, image) => {
    if (err) throw err;
        image.write('kyc_step2.png');
    });*/
    
    window.cv.cvtColor(pImage, pImage, window.cv.COLOR_RGBA2GRAY);
    console.log(pImage);
    let probabilities=await classifyDocument(pImage,model);
    //console.log("IDENTIFIED DOCUMENT: "+docclass)
    return probabilities
}

export default kycimage;
