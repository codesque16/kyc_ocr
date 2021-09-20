(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{103:function(e,t,a){e.exports=a(187)},109:function(e,t,a){},112:function(e,t,a){},120:function(e,t){},121:function(e,t){},122:function(e,t){},123:function(e,t){},124:function(e,t){},125:function(e,t){},126:function(e,t,a){},182:function(e,t,a){},185:function(e,t,a){},187:function(e,t,a){"use strict";a.r(t);var n=a(3),o=a(4),r=a(6),s=a(7),i=a(1),l=a.n(i),c=a(38),d=a.n(c),u=a(36),p=(a(108),a(109),a(12)),m=a(191),h=a(194),f=a(100),v=a(195),w=function(e){Object(r.a)(a,e);var t=Object(s.a)(a);function a(){return Object(n.a)(this,a),t.apply(this,arguments)}return Object(o.a)(a,[{key:"render",value:function(){return this.props.show?l.a.createElement(v.a,{variant:this.props.variant||"danger",onClose:this.props.onClose,dismissible:!0},this.props.title&&l.a.createElement("h5",null,l.a.createElement("strong",null,this.props.title)),l.a.createElement("div",{dangerouslySetInnerHTML:{__html:this.props.message}})):null}}]),a}(i.Component),g=(a(112),function(e){Object(r.a)(a,e);var t=Object(s.a)(a);function a(){return Object(n.a)(this,a),t.apply(this,arguments)}return Object(o.a)(a,[{key:"render",value:function(){return l.a.createElement("div",{className:"About container"},l.a.createElement("h1",null,"About"),l.a.createElement("p",null,"This is a TensorFlow.js web application where users can classify images selected locally or taken with their device's camera. The app uses TensorFlow.js and a pre-trained model converted to the TensorFlow.js format to provide the inference capabilities. This model is saved locally in the browser using IndexedDB. A service worker is also used to provide offline capabilities."))}}]),a}(i.Component)),b=a(11),E=a.n(b),y=a(15),x=a(190),O=a(189),C=a(96),A=a(196),k=a(192),j=a(193),S=a(197),U=a(62),N=a(101),I=a(87),L=a.n(I),M=a(16),T=a(102),R=["isLoading","text","loadingText","className","disabled"],D=function(e){var t=e.isLoading,a=e.text,n=e.loadingText,o=e.className,r=void 0===o?"":o,s=e.disabled,i=void 0!==s&&s,c=Object(T.a)(e,R);return l.a.createElement(O.a,Object.assign({className:"LoadButton ".concat(r),disabled:i||t},c),t&&l.a.createElement(x.a,{as:"span",animation:"border",size:"sm",role:"status","aria-hidden":"true"})," ",t?n:a)},P={0:"AADHAAR",1:"DRIVING LICENSE",2:"PAN",3:"PASSPORT",4:"VOTER ID"},_={API_ENDPOINT:"http://localhost:5000/api"},W=(a(126),a(127),a(186).Image);function F(e,t,a){var n=e.cols,o=e.rows,r=parseInt(n*Math.min(t/n,a/o)),s=parseInt(o*Math.min(t/n,a/o)),i=new window.cv.Mat,l=new window.cv.Size(r,s);window.cv.resize(e,i,l,0,0,window.cv.INTER_CUBIC);for(var c=window.cv.Mat.zeros(a,t,window.cv.CV_8U),d=~~((t-r)/2),u=~~((a-s)/2),p=0;p<s;p++)for(var m=0;m<r;m++)c.ucharPtr(u+p,d+m)[0]=i.ucharPtr(p,m)[0];return c}function B(e,t){return G.apply(this,arguments)}function G(){return(G=Object(y.a)(E.a.mark(function e(t,a){var n,o;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.t0=console,e.next=3,M.g(t.data).div(M.e(255)).array();case 3:return e.t1=e.sent,e.t0.log.call(e.t0,e.t1),e.next=7,a.predict([M.g(t.data).div(M.e(255)).reshape([1,80,80,1])]);case 7:return n=e.sent,e.next=10,n.array();case 10:return o=e.sent,e.abrupt("return",M.f(o[0]).array());case 12:case"end":return e.stop()}},e)}))).apply(this,arguments)}function z(e){console.log(e),window.cv.cvtColor(e,e,window.cv.COLOR_RGBA2GRAY),console.log(e);var t=new window.cv.Mat;console.log(t);var a=window.cv.Mat.ones(9,9,window.cv.CV_8U);return console.log(a),window.cv.morphologyEx(e,t,window.cv.MORPH_BLACKHAT,a),window.cv.cvtColor(t,t,window.cv.COLOR_GRAY2RGBA),console.log(t),t}function K(e){return H.apply(this,arguments)}function H(){return(H=Object(y.a)(E.a.mark(function e(t){var a;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,W.load(t);case 2:return a=e.sent,e.abrupt("return",a);case 4:case"end":return e.stop()}},e)}))).apply(this,arguments)}var Y=function(){var e=Object(y.a)(E.a.mark(function e(t,a){var n,o,r,s,i;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,K(t);case 2:return n=e.sent,e.next=5,new window.cv.matFromImageData(n);case 5:return o=e.sent,console.log(o.rows),r=z(o),console.log(r),window.cv.cvtColor(r,r,window.cv.COLOR_RGBA2GRAY),console.log(r),s=F(r,80,80),console.log(s),window.cv.cvtColor(s,s,window.cv.COLOR_GRAY2RGBA),window.cv.cvtColor(s,s,window.cv.COLOR_RGBA2GRAY),console.log(s),e.next=18,B(s,a);case 18:return i=e.sent,e.abrupt("return",i);case 20:case"end":return e.stop()}},e)}));return function(t,a){return e.apply(this,arguments)}}(),V="model/model.json",J=224,$=224,q=5,Q="web-model",X=function(e){Object(r.a)(a,e);var t=Object(s.a)(a);function a(e){var o;return Object(n.a)(this,a),(o=t.call(this,e)).initWebcam=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,M.b.webcam(o.refs.webcam,{resizeWidth:$,resizeHeight:$,facingMode:"environment"});case 3:o.webcam=e.sent,e.next=9;break;case 6:e.prev=6,e.t0=e.catch(0),o.refs.noWebcam.style.display="block";case 9:case"end":return e.stop()}},e,null,[[0,6]])})),o.startWebcam=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:o.webcam&&o.webcam.start();case 1:case"end":return e.stop()}},e)})),o.stopWebcam=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:o.webcam&&o.webcam.stop();case 1:case"end":return e.stop()}},e)})),o.getModelInfo=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,fetch("".concat(_.API_ENDPOINT,"/model_info"),{method:"GET"}).then(function(){var e=Object(y.a)(E.a.mark(function e(t){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.json().then(function(e){o.modelLastUpdated=e.last_updated}).catch(function(e){console.log("Unable to get parse model info.")});case 2:case"end":return e.stop()}},e)}));return function(t){return e.apply(this,arguments)}}()).catch(function(e){console.log("Unable to get model info")});case 2:case"end":return e.stop()}},e)})),o.updateModel=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("Updating the model: "+Q),o.setState({isDownloadingModel:!0}),e.next=4,M.d(V);case 4:return o.model=e.sent,e.next=7,o.model.save("indexeddb://"+Q);case 7:o.setState({isDownloadingModel:!1,modelUpdateAvailable:!1,showModelUpdateAlert:!1,showModelUpdateSuccess:!0});case 8:case"end":return e.stop()}},e)})),o.classifyLocalImage=Object(y.a)(E.a.mark(function e(){var t,a,n,r,s,i,l,c,d,u,p;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return o.setState({isClassifying:!0}),t=o.refs.cropper.getCroppedCanvas(),console.log(t),console.log(M.a.fromPixels(t)),a=M.h(function(){return M.a.fromPixels(t).toFloat()}),console.log(t.toDataURL()),e.next=8,new Uint8ClampedArray(a.array());case 8:return n=e.sent,console.log(n),r=t.toDataURL(),s={method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({imagebuf:r,height:a.shape[0],width:a.shape[1],channels:a.shape[2]})},console.log(s),e.next=15,Y(r,o.model);case 15:return i=e.sent,console.log(i),e.next=19,o.getTopKClasses(i,q);case 19:l=e.sent,o.setState({predictions:l,isClassifying:!1,photoSettingsOpen:!o.state.photoSettingsOpen}),c=o.refs.canvas.getContext("2d"),d=$/t.width,u=$/t.height,p=Math.min(d,u),c.clearRect(0,0,$,$),c.drawImage(t,0,0,t.width*p,t.height*p),a.dispose();case 28:case"end":return e.stop()}},e)})),o.classifyWebcamImage=Object(y.a)(E.a.mark(function e(){var t,a,n,r,s,i,l;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return o.setState({isClassifying:!0}),e.next=3,o.webcam.capture();case 3:return t=e.sent,console.log(t.print()),a=M.c.resizeBilinear(t,[J,J]),e.next=8,o.processImage(a);case 8:return n=e.sent,r=o.model.predict(n),e.next=12,r.data();case 12:return s=e.sent,e.next=15,o.getTopKClasses(s,q);case 15:return i=e.sent,o.setState({predictions:i,isClassifying:!1,photoSettingsOpen:!o.state.photoSettingsOpen}),l=M.h(function(){return t.toFloat().div(255)}),e.next=20,M.a.toPixels(l,o.refs.canvas);case 20:t.dispose(),r.dispose(),l.dispose(),n.dispose();case 24:case"end":return e.stop()}},e)})),o.processImage=function(){var e=Object(y.a)(E.a.mark(function e(t){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return console.log(t.expandDims(0).toFloat()),e.abrupt("return",M.h(function(){return t.expandDims(0).toFloat().div(127).sub(1)}));case 2:case"end":return e.stop()}},e)}));return function(t){return e.apply(this,arguments)}}(),o.getTopKClasses=function(){var e=Object(y.a)(E.a.mark(function e(t,a){var n,o,r,s,i,l,c;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:for(n=[],o=0;o<t.length;o++)n.push({value:t[o],index:o});for(n.sort(function(e,t){return t.value-e.value}),r=new Float32Array(a),s=new Int32Array(a),i=0;i<a;i++)r[i]=n[i].value,s[i]=n[i].index;for(l=[],c=0;c<s.length;c++)l.push({className:P[s[c]],probability:(100*r[c]).toFixed(2)});return e.abrupt("return",l);case 9:case"end":return e.stop()}},e)}));return function(t,a){return e.apply(this,arguments)}}(),o.handlePanelClick=function(e){o.setState({photoSettingsOpen:!o.state.photoSettingsOpen})},o.handleFileChange=function(e){e.target.files&&e.target.files.length>0&&o.setState({file:URL.createObjectURL(e.target.files[0]),filename:e.target.files[0].name})},o.handleTabSelect=function(e){switch(e){case"camera":o.startWebcam();break;case"localfile":o.setState({filename:null,file:null}),o.stopWebcam()}},o.webcam=null,o.model=null,o.modelLastUpdated=null,o.state={modelLoaded:!1,filename:"",isModelLoading:!1,isClassifying:!1,predictions:[],photoSettingsOpen:!0,modelUpdateAvailable:!1,showModelUpdateAlert:!1,showModelUpdateSuccess:!1,isDownloadingModel:!1},o}return Object(o.a)(a,[{key:"componentDidMount",value:function(){var e=Object(y.a)(E.a.mark(function e(){var t,a,n;return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:if(!("indexedDB"in window)){e.next=36;break}return e.prev=1,e.next=4,M.d("indexeddb://"+Q);case 4:return this.model=e.sent,e.prev=5,e.next=8,Object(N.a)("tensorflowjs",1);case 8:return t=e.sent,e.next=11,t.transaction("model_info_store").objectStore("model_info_store").get(Q);case 11:return a=e.sent,n=new Date(a.modelArtifactsInfo.dateSaved),e.next=15,this.getModelInfo();case 15:console.log(this.modelLastUpdated),!this.modelLastUpdated||n>=new Date(this.modelLastUpdated).getTime()?console.log("Using saved model"):this.setState({modelUpdateAvailable:!0,showModelUpdateAlert:!0}),e.next=23;break;case 19:e.prev=19,e.t0=e.catch(5),console.warn(e.t0),console.warn("Could not retrieve when model was saved.");case 23:e.next=34;break;case 25:return e.prev=25,e.t1=e.catch(1),console.log("Not found in IndexedDB. Loading and saving..."),console.log(e.t1),e.next=31,M.d(V);case 31:return this.model=e.sent,e.next=34,this.model.save("indexeddb://"+Q);case 34:e.next=40;break;case 36:return console.warn("IndexedDB not supported."),e.next=39,M.d(V);case 39:this.model=e.sent;case 40:this.setState({modelLoaded:!0}),this.initWebcam();case 42:case"end":return e.stop()}},e,this,[[1,25],[5,19]])}));return function(){return e.apply(this,arguments)}}()},{key:"componentWillUnmount",value:function(){var e=Object(y.a)(E.a.mark(function e(){return E.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:this.webcam&&this.webcam.stop();try{this.model.dispose()}catch(t){}case 2:case"end":return e.stop()}},e,this)}));return function(){return e.apply(this,arguments)}}()},{key:"render",value:function(){var e=this;return l.a.createElement("div",{className:"Classify container"},!this.state.modelLoaded&&l.a.createElement(i.Fragment,null,l.a.createElement(x.a,{animation:"border",role:"status"},l.a.createElement("span",{className:"sr-only"},"Loading..."))," ",l.a.createElement("span",{className:"loading-model-text"},"Loading Model")),this.state.modelLoaded&&l.a.createElement(i.Fragment,null,l.a.createElement(O.a,{onClick:this.handlePanelClick,className:"classify-panel-header","aria-controls":"photo-selection-pane","aria-expanded":this.state.photoSettingsOpen},"Take or Select a Photo to Classify",l.a.createElement("span",{className:"panel-arrow"},this.state.photoSettingsOpen?l.a.createElement(U.b,null):l.a.createElement(U.c,null))),l.a.createElement(C.a,{in:this.state.photoSettingsOpen},l.a.createElement("div",{id:"photo-selection-pane"},this.state.modelUpdateAvailable&&this.state.showModelUpdateAlert&&l.a.createElement(m.a,null,l.a.createElement(v.a,{variant:"info",show:this.state.modelUpdateAvailable&&this.state.showModelUpdateAlert,onClose:function(){return e.setState({showModelUpdateAlert:!1})},dismissible:!0},"An update for the ",l.a.createElement("strong",null,this.state.modelType)," model is available.",l.a.createElement("div",{className:"d-flex justify-content-center pt-1"},!this.state.isDownloadingModel&&l.a.createElement(O.a,{onClick:this.updateModel,variant:"outline-info"},"Update"),this.state.isDownloadingModel&&l.a.createElement("div",null,l.a.createElement(x.a,{animation:"border",role:"status",size:"sm"},l.a.createElement("span",{className:"sr-only"},"Downloading..."))," ",l.a.createElement("strong",null,"Downloading..."))))),this.state.showModelUpdateSuccess&&l.a.createElement(m.a,null,l.a.createElement(v.a,{variant:"success",onClose:function(){return e.setState({showModelUpdateSuccess:!1})},dismissible:!0},"The ",l.a.createElement("strong",null,this.state.modelType)," model has been updated!")),l.a.createElement(A.a,{defaultActiveKey:"camera",id:"input-options",onSelect:this.handleTabSelect,className:"justify-content-center"},l.a.createElement(k.a,{eventKey:"camera",title:"Take Photo"},l.a.createElement("div",{id:"no-webcam",ref:"noWebcam"},l.a.createElement("span",{className:"camera-icon"},l.a.createElement(U.a,null)),"No camera found. ",l.a.createElement("br",null),"Please use a device with a camera, or upload an image instead."),l.a.createElement("div",{className:"webcam-box-outer"},l.a.createElement("div",{className:"webcam-box-inner"},l.a.createElement("video",{ref:"webcam",autoPlay:!0,playsInline:!0,muted:!0,id:"webcam",width:"448",height:"448"}))),l.a.createElement("div",{className:"button-container"},l.a.createElement(D,{variant:"primary",size:"lg",onClick:this.classifyWebcamImage,isLoading:this.state.isClassifying,text:"Classify",loadingText:"Classifying..."}))),l.a.createElement(k.a,{eventKey:"localfile",title:"Select Local File"},l.a.createElement(j.a.Group,{controlId:"file"},l.a.createElement(j.a.Label,null,"Select Image File"),l.a.createElement("br",null),l.a.createElement(j.a.Label,{className:"imagelabel"},this.state.filename?this.state.filename:"Browse..."),l.a.createElement(j.a.Control,{onChange:this.handleFileChange,type:"file",accept:"image/*",className:"imagefile"})),this.state.file&&l.a.createElement(i.Fragment,null,l.a.createElement("div",{id:"local-image"},l.a.createElement(L.a,{ref:"cropper",src:this.state.file,style:{height:"100%",width:"100%"},guides:!0,viewMode:2})),l.a.createElement("div",{className:"button-container"},l.a.createElement(D,{variant:"primary",size:"lg",disabled:!this.state.filename,onClick:this.classifyLocalImage,isLoading:this.state.isClassifying,text:"Classify",loadingText:"Classifying..."}))))))),this.state.predictions.length>0&&l.a.createElement("div",{className:"classification-results"},l.a.createElement("h3",null,"Predictions"),l.a.createElement("canvas",{ref:"canvas",width:$,height:$}),l.a.createElement("br",null),l.a.createElement(S.a,null,this.state.predictions.map(function(e){return l.a.createElement(S.a.Item,{key:e.className},l.a.createElement("strong",null,e.className)," ",e.probability,"%")})))))}}]),a}(i.Component),Z=(a(182),function(){return l.a.createElement("div",{className:"NotFound"},l.a.createElement("h1",null,"404"),l.a.createElement("h3",null,"The page you were looking for is not here."),l.a.createElement("a",{href:"/"},"Go Home"))}),ee=function(e){var t=e.childProps;return l.a.createElement(p.c,null,l.a.createElement(p.a,{path:"/",exact:!0,component:X,props:t}),l.a.createElement(p.a,{path:"/about",exact:!0,component:g,props:t}),l.a.createElement(p.a,{component:Z}))},te=(a(185),a(99)),ae=function(e){Object(r.a)(a,e);var t=Object(s.a)(a);function a(e){var o;Object(n.a)(this,a),(o=t.call(this,e)).dismissUpdateAlert=function(e){o.setState({showUpdateAlert:!1})};return o.state={showUpdateAlert:!0,reloadMsg:"\n      New content is available.<br />\n      Please <a href='javascript:location.reload();'>reload</a>.<br />\n      <small>If reloading doesn't work, close all tabs/windows of this web application,\n      and then reopen the application.</small>\n    "},o}return Object(o.a)(a,[{key:"componentDidMount",value:function(){Object(te.a)({onLoaded:function(){console.log(window.cv.getBuildInformation()),console.log("open cv loaded")},onFailed:function(){return console.log("open cv failed to load")},version:"4.5.1"})}},{key:"render",value:function(){return l.a.createElement("div",{className:"App"},l.a.createElement(m.a,null,l.a.createElement(h.a,{collapseOnSelect:!0,className:"app-nav-bar",variant:"dark",expand:"lg"},l.a.createElement(h.a.Brand,{href:"/"},"KYC"),l.a.createElement(h.a.Toggle,{"aria-controls":"basic-navbar-nav"}),l.a.createElement(h.a.Collapse,{id:"basic-navbar-nav"},l.a.createElement(f.a,{className:""},l.a.createElement(u.b,{className:"nav-link",to:"/"},"Classify"),l.a.createElement(u.b,{className:"nav-link",to:"/about"},"About")))),this.props.updateAvailable&&this.state.showUpdateAlert&&l.a.createElement("div",{style:{paddingTop:"10px"}},l.a.createElement(w,{title:"",variant:"info",message:this.state.reloadMsg,show:this.props.updateAvailable&&this.state.showUpdateAlert,onClose:this.dismissUpdateAlert}))),l.a.createElement(m.a,null,l.a.createElement(ee,null)))}}]),a}(i.Component),ne=Object(p.f)(ae),oe=Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));function re(e){if("serviceWorker"in navigator){if(new URL("/kyc_ocr",window.location.href).origin!==window.location.origin)return;window.addEventListener("load",function(){var t="".concat("/kyc_ocr","/service-worker.js");oe?(!function(e,t){fetch(e).then(function(a){var n=a.headers.get("content-type");404===a.status||null!=n&&-1===n.indexOf("javascript")?navigator.serviceWorker.ready.then(function(e){e.unregister().then(function(){window.location.reload()})}):se(e,t)}).catch(function(){console.log("No internet connection found. App is running in offline mode.")})}(t,e),navigator.serviceWorker.ready.then(function(){console.log("This web app is being served cache-first by a service worker. To learn more, visit https://bit.ly/CRA-PWA")})):se(t,e)})}}function se(e,t){navigator.serviceWorker.register(e).then(function(e){function a(a){console.log("New content is available and will be used when all tabs for this page are closed. See http://bit.ly/CRA2-PWA."),t&&t.onUpdate&&t.onUpdate(e,a)}e.waiting&&e.active&&a(e.waiting),e.onupdatefound=function(){var n=e.installing;null!=n&&(n.onstatechange=function(){"installed"===n.state&&(navigator.serviceWorker.controller?a(n):(console.log("Content is cached for offline use."),t&&t.onSuccess&&t.onSuccess(e)))})}}).catch(function(e){console.error("Error during service worker registration:",e)})}var ie=function(e){Object(r.a)(a,e);var t=Object(s.a)(a);function a(){var e;Object(n.a)(this,a);for(var o=arguments.length,r=new Array(o),s=0;s<o;s++)r[s]=arguments[s];return(e=t.call.apply(t,[this].concat(r))).state={contentCached:!1,updateAvailable:!1},e.handleUpdate=function(t){var a=t.waiting;a&&a.postMessage({type:"SKIP_WAITING"}),e.setState({updateAvailable:!0})},e}return Object(o.a)(a,[{key:"componentDidMount",value:function(){re({onUpdate:this.handleUpdate})}},{key:"render",value:function(){return l.a.createElement(u.a,null,l.a.createElement(ne,{updateAvailable:this.state.updateAvailable}))}}]),a}(i.Component);d.a.render(l.a.createElement(ie,null),document.getElementById("root"))}},[[103,1,2]]]);
//# sourceMappingURL=main.98290182.chunk.js.map