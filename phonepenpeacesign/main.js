// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

// names for classes
const CLASS_NAMES = ['phone', 'pen', 'peace sign']

class Main {
  constructor(){
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.knn = new KNNImageClassifier(NUM_CLASSES, TOPK);

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add container element to body
    const container = document.createElement('div');
    container.className = 'container'
    document.body.appendChild(container);

    // Add video element to DOM
    const left = document.createElement('div');
    left.className = 'left'
    left.appendChild(this.video);
    container.appendChild(left);

    // Create training buttons and info texts
    const right = document.createElement('div');
    right.className = 'right'
    this.write_button(right, "📱", 0);
    this.write_button(right, "🖊️", 1);
    this.write_button(right, "✌️", 2);

    container.appendChild(right);

    // prediction text
    const div = document.createElement('div');
    document.body.appendChild(div);
    const predText = document.createElement('span')
    predText.innerText = '';
    div.appendChild(predText);
    this.prediction = predText;


    // Setup webcam
    navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then((stream) => {
      this.video.srcObject = stream;
      this.video.width = IMAGE_SIZE;
      this.video.height = IMAGE_SIZE;

      this.video.addEventListener('playing', ()=> this.videoPlaying = true);
      this.video.addEventListener('paused', ()=> this.videoPlaying = false);
    })

    // Load knn model
    this.knn.load()
    .then(() => this.start());
  }

  write_button(right, button_text, i){
    const div = document.createElement('div');
    right.appendChild(div);
    div.style.marginBottom = '10px';

    // Create training button
    const button = document.createElement('button')
    button.innerText = button_text;
    div.appendChild(button);

    // Listen for mouse events when clicking the button
    button.addEventListener('mousedown', () => this.training = i);
    button.addEventListener('mouseup', () => this.training = -1);

    // Create info text
    const infoText = document.createElement('span')
    infoText.innerText = " No examples added";
    div.appendChild(infoText);
    this.infoTexts.push(infoText);
  }

  start(){
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop(){
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  animate(){
    if(this.videoPlaying){
      // Get image data from video element
      const image = dl.fromPixels(this.video);

      // Train class if one of the buttons is held down
      if(this.training != -1){
        // Add current image to classifier
        this.knn.addImage(image, this.training)
      }

      // If any examples have been added, run predict
      const exampleCount = this.knn.getClassExampleCount();
      if(Math.max(...exampleCount) > 0){
        this.knn.predictClass(image)
        .then((res)=>{
          for(let i=0;i<NUM_CLASSES; i++){
            // Update info text
            if(exampleCount[i] > 0){
              this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`
            }
          }
          // Update prediction text
          if(res.confidences[res.classIndex] > 0.7){
            this.prediction.innerText = 'That looks like a ' + CLASS_NAMES[res.classIndex];
          } else {
            this.prediction.innerText = "I'm not sure what that is..."
          }
        })
        // Dispose image when done
        .then(()=> image.dispose())
      } else {
        image.dispose()
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));


  }
}

window.addEventListener('load', () => new Main());
