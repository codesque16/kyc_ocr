//const { cv, imdecode,cvTranslateError } = require('opencv-wasm');
//const {imdecode} = require('./opencv.js');
//const cv = require("@techstark/opencv-js");
//const cv4node = require('opencv4nodejs');
const { createWorker } = require('tesseract.js');
const PSM = require('tesseract.js/src/constants/PSM.js')
const OEM = require('tesseract.js/src/constants/OEM.js')
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');
const { Image } = require('image-js');
//const bufferImage = require("buffer-image");
const encode = require('image-encode')
//require('@tensorflow/tfjs-node');

const worker = createWorker({
  langPath: path.join(__dirname, 'public', 'tesseract_data'),
  logger: m => console.log(m),
});

function preprocessImage(img,w,h) {
    let img_w=img.cols
    let img_h=img.rows
    let new_w = parseInt(img_w * Math.min(w/img_w, h/img_h))
    let new_h = parseInt(img_h * Math.min(w/img_w, h/img_h))
    //console.log(new_w,new_h)

    let dst = new cv.Mat();
    let dsize = new cv.Size(new_w,new_h);
    //console.log(dsize)
    //console.log(img_w,img_h,img.data)
    // You can try more different parameters
    cv.resize(img, dst, dsize, 0, 0, cv.INTER_CUBIC);
    let canvas = cv.Mat.zeros(h,w, cv.CV_8U);
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

async function classifyDocument(preprocessed_img) {
  const classes=['AADHAAR_EXTRACTION', 'DL_EXTRACTION', 'PAN_EXTRACTION',
  'PASSPORT_EXTRACTION', 'VOTER_EXTRACTION']
  var predicted="NO_OUTPUT";
  let model = await tf.loadLayersModel('file://public/kyc_model/model.json');
  let scores = await model.predict([tf.tensor(preprocessed_img.data).div(tf.scalar(255.0)).reshape([1, 80, 80, 1])]);
  //console.log(await scores.data())
  let score_array= await scores.array()
  predicted=score_array[0].indexOf(Math.max(...score_array));
  //console.log(classes[predicted])
  //  await tf.loadLayersModel('file://public/kyc_model/model.json').then(async function(model) {
  //    await model.predict([tf.tensor(preprocessed_img.data).reshape([1, 80, 80, 1])]).array().then(function(scores){
  //      scores = scores[0];
  //      predicted = scores.indexOf(Math.max(...scores));
  //      console.log(classes[predicted])
  //    });
  //});

  //console.log(scores)
  return tf.softmax(score_array[0]).array()
}

async function getTextFromImage(buffer) {
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

function getBlackHatImage(imageMat) {
  cv.cvtColor(imageMat, imageMat, cv.COLOR_RGBA2GRAY);
  let dst = new cv.Mat();
  let M = cv.Mat.ones(9,9, cv.CV_8U);
  cv.morphologyEx(imageMat, dst, cv.MORPH_BLACKHAT, M);
  //console.log(dst.cols)
  //console.log(dst.rows)
  //console.log(dst.data)
  cv.cvtColor(dst, dst, cv.COLOR_GRAY2RGBA);

  /*new Jimp({ data: dst.data, width: dst.cols, height: dst.rows }, (err, image) => {
  if (err) throw err;
  image.write('tesseract.png');
  // this image is 1280 x 768, pixels are loaded from the given buffer.
  });
  */
  return dst
}


async function getImage(buffer) {
  let img = await Image.load(buffer);
//  let output = tf.browser.fromPixels(img);
  return img;
}

exports.kycimage = async function kycimage(buffer,h,w) {
    console.log(typeof buffer);
    console.log(new cv.Mat());
    console.log(cv.getBuildInformation());
    //const regex = /data:image.*?base64,\s*/i;
    //buffer=buffer.replace(regex,'')
    //buff = Buffer.from(buffer,'base64');
    //buff=buff.decodeArrayBuffer();
    let jsimg = await getImage(buffer);
    console.log(jsimg);
    const src = await new cv.matFromImageData(jsimg)
    src.convertTo(src, cv.CV_8U);
    //console.log(src.rows)
    //console.log(src.cols)
    //console.log(src.data)
    new Jimp({ data: src.data, width: src.cols, height: src.rows }, (err, image) => {
          if (err) throw err;
          image.write('kyc_step1.png');
    });
    let dst = getBlackHatImage(src);
    //new Jimp({ data: dst.data, width: dst.cols, height: dst.rows }, (err, image) => {
    //    if (err) throw err;
    //    image.write('tesseract.png');
    //});
    //let tesbase64 = 'data:image/png;base64,' + Buffer.from(dst.data,'binary').toString('base64');
    //console.log(await im2base64(Buffer.from(src.data)));
    //console.log(encode(src.data,[src.cols,src.rows],'png'))
    //await getTextFromImage('tesseract.png')
    //    .then(console.log);

    cv.cvtColor(dst, dst, cv.COLOR_RGBA2GRAY);
    let pImage= preprocessImage(dst,80,80);
    cv.cvtColor(pImage, pImage, cv.COLOR_GRAY2RGBA);
    new Jimp({ data: pImage.data, width: pImage.cols, height: pImage.rows }, (err, image) => {
    if (err) throw err;
        image.write('kyc_step2.png');
    });
    
    cv.cvtColor(pImage, pImage, cv.COLOR_RGBA2GRAY);
    let probabilities=await classifyDocument(pImage);
    //console.log("IDENTIFIED DOCUMENT: "+docclass)
    return probabilities
}
