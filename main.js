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

import "@babel/polyfill";
import * as mobilenetModule from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

const classes = ["Touching", "Not Touching"];
// Number of classes to classify
const NUM_CLASSES = classes.length;

let testPrediction = false;
let knn;
let touchingCount = 0;
let notTouchingCount = 0;
let lastSoundPlay = new Date();

const trainingSection = document.getElementsByClassName("training-section")[0];
const videoArea = document.getElementsByClassName("video-area")[0];
const buttonsSection = document.createElement("section");
const statusText = document.getElementsByClassName("status")[0];
buttonsSection.classList.add("buttons-section");

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement("video");
    this.video.classList.add("video");
    this.video.setAttribute("autoplay", "");
    this.video.setAttribute("playsinline", "");

    // Add video element to DOM
    videoArea.appendChild(this.video);

    // Create training buttons and info texts
    for (let i = 0; i < NUM_CLASSES; i++) {
      const buttonBlock = document.createElement("div");
      buttonBlock.classList.add("button-block");

      // Create training button
      const button = document.createElement("button");
      button.innerText = classes[i];
      buttonBlock.appendChild(button);
      buttonsSection.appendChild(buttonBlock);

      const div = document.createElement("div");
      div.classList.add("examples-text");
      buttonBlock.appendChild(div);
      div.style.marginBottom = "10px";

      // Listen for mouse events when clicking the button
      button.addEventListener("mousedown", () => {
        testPrediction = false;
        this.training = i;
      });
      button.addEventListener("touchstart", () => {
        testPrediction = false;
        this.training = i;
      });
      button.addEventListener("mouseup", () => {
        testPrediction = true;
        this.training = -1;
      });
      button.addEventListener("touchend", () => {
        testPrediction = true;
        this.training = -1;
      });

      // // Create info text
      const infoText = document.createElement("span");
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }
    trainingSection.appendChild(buttonsSection);

    // Setup webcam
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(stream => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener(
          "playing",
          () => (this.videoPlaying = true)
        );
        this.video.addEventListener(
          "paused",
          () => (this.videoPlaying = false)
        );
      });
  }

  async bindPage() {
    knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    const stateJson = await fetch("state.json");
    const { state } = await stateJson.json();
    knn.setClassifierDataset(
      Object.fromEntries(
        state.map(([label, data, shape]) => [label, tf.tensor(data, shape)])
      )
    );
    testPrediction = true;

    trainingSection.style.display = "block";

    setInterval(statusCheck, 200);

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.browser.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, "conv_preds");

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        knn.addExample(logits, this.training);
      }

      const numClasses = knn.getNumClasses();

      //start prediction
      if (testPrediction && numClasses > 0) {
        // If classes have been added run predict
        logits = infer();
        const res = await knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {
          // The number of examples for each class
          const exampleCount = knn.getClassExampleCount();

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res
              .confidences[i] * 100}%`;
          }
        }

        if (res.confidences[0] > 0.5) {
          touchingCount++;
        } else {
          notTouchingCount++;
        }
      } else {
        // The number of examples for each class
        const exampleCount = knn.getClassExampleCount();

        for (let i = 0; i < NUM_CLASSES; i++) {
          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`;
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

function statusCheck() {
  if (touchingCount > notTouchingCount) {
    statusText.innerText = "Stop touching your face!";
    statusText.classList.add("status-touching");
    statusText.classList.remove("status-not-touching");
    buzz();
  } else {
    statusText.innerText = "Good, you are not touching your face!";
    statusText.classList.remove("status-touching");
    statusText.classList.add("status-not-touching");
  }
  touchingCount = 0;
  notTouchingCount = 0;
}

function buzz() {
  if (new Date() - lastSoundPlay > 2000) {
    let sound = new Audio("wrong.mp3"); // buffers automatically when created
    sound.play();
    lastSoundPlay = new Date();
  }
}

window.addEventListener("load", () => new Main());

window.saveDataset = () => {
  let state = JSON.stringify(
    Object.entries(knn.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape
    ])
  );
  localStorage.setItem("state", state);
};
