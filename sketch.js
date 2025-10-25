let video;
let handPose;
let hands = [];

// --- TEACAHBLE MACHINE CONFIG ---
const TM_URL = "./my_model/"; // Ensure your model files (model.json, metadata.json) are here
let tmModel, maxPredictions;
let isModelReady = false;
let lastPredictionTime = 0;
const PREDICTION_COOLDOWN = 1500; // 1.5 seconds cooldown to prevent rapid triggers
// --------------------------------

const animals = ["deer", "cat", "dog", "snake", "elephant", "flower", "rabbit", "horse", "fish", "bird"];
let animalDetected = null;

const popup = document.getElementById("popup");
const closeBtn = document.getElementById("close");
const postTitle = document.getElementById("post-title");
const postBody = document.getElementById("post-body");
const postImages = document.getElementById("post-images");

// NOTE: You must include the Teachable Machine library in your HTML:
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
// <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

function preload() {
    // We still load handPose to get the landmarks for drawing/visuals if needed
    handPose = ml5.handPose();
}

async function setup() {
    background('#000032');
    // Set a fixed size for the canvas where the video will be drawn
    const videoWidth = 400;
    const videoHeight = 300;
    
    // Check if createCanvas is available (p5.js setup)
    if (typeof createCanvas !== 'undefined') {
        const canvas = createCanvas(videoWidth, videoHeight);
        canvas.parent("sketch-canvas");
    }

    video = createCapture(VIDEO);
    video.size(videoWidth, videoHeight);
    
    video.elt.style.filter = 'grayscale(100%) contrast(150%)';
    video.elt.style.transform = 'scaleX(-1)';
    
    // We start the HandPose detection to get landmarks for drawing later
    handPose.detectStart(video, gotHands);
    
    // --- LOAD TEACHABLE MACHINE MODEL ---
    // Added check for tmImage global variable to provide a clearer error message
    // if the required script is missing from the HTML.
    if (typeof tmImage === 'undefined') {
        console.error("FATAL ERROR: tmImage is not defined. Please ensure you have included the Teachable Machine library scripts in your HTML file as noted in the comments.");
        return; // Stop setup execution if the core library is missing
    }

    try {
        const modelURL = TM_URL + "model.json";
        const metadataURL = TM_URL + "metadata.json";
        tmModel = await tmImage.load(modelURL, metadataURL);
        maxPredictions = tmModel.getTotalClasses();
        isModelReady = true;
        console.log("Teachable Machine model loaded successfully!");
        modelReady(); // Call the standard p5/ml5 readiness function
    } catch (error) {
        // Catches errors related to model loading (e.g., file not found)
        console.error("Failed to load Teachable Machine model:", error);
    }
}

function modelReady() {
    console.log("HandPose and Teachable Machine models ready!");
}

function gotHands(results) {
    hands = results;
}

function draw() {
    // The p5.js draw loop runs constantly
    image(video, 0, 0, width, height);

    // Call the prediction function to run the TM model
    if (isModelReady) {
        detectGesture();
    }
    
    // Optional: Draw the hand landmarks if needed for debugging or visual flair
    // drawLandmarks(); 
}

/* function drawLandmarks() {
    for (let i = 0; i < hands.length; i++) {
        const hand = hands[i];
        for (let j = 0; j < hand.landmarks.length; j++) {
            const [x, y, z] = hand.landmarks[j];
            fill(255, 0, 0);
            ellipse(x, y, 10, 10);
        }
    }
} */


async function detectGesture() {
    // Use the TM model on the video element's canvas
    const now = Date.now();
    if (!isModelReady || now - lastPredictionTime < PREDICTION_COOLDOWN) {
        return; // Skip prediction if model isn't ready or cooldown is active
    }

    // Pass the raw video element (or its internal canvas) to the TM model
    const prediction = await tmModel.predict(video.elt);
    
    let highestProb = 0;
    let predictedClass = "Background"; 
    
    // Find the class with the highest confidence
    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction = prediction[i];
        
        // Console output for debugging
        // console.log(`${classPrediction.className}: ${classPrediction.probability.toFixed(2)}`);

        if (classPrediction.probability > highestProb) {
            highestProb = classPrediction.probability;
            predictedClass = classPrediction.className;
        }
    }

    const MIN_CONFIDENCE = 0.85; // Require 85% confidence to trigger a gesture
    
    if (highestProb > MIN_CONFIDENCE && predictedClass !== "Background") {
        lastPredictionTime = now; // Reset cooldown timer
        // The detected class name should correspond to one of your animals (e.g., "cat", "dog", "snake")
        showAnimal(predictedClass.toLowerCase()); 
        console.log(`Detected: ${predictedClass} with ${highestProb.toFixed(2)} confidence.`);
    }
}


async function showAnimal(animal) {
    // Check if the detected animal is in our valid list (animals array)
    if (!animals.includes(animal)) {
        console.log(`TM detected "${animal}", but it's not a valid dream animal. Ignoring.`);
        return;
    }

    if (animalDetected === animal) return; 
    animalDetected = animal;
    
    // Ensure the main UI reflects the last detected animal for the user
    console.log(`Triggering dream for: ${animal}`);

    try {
        // Fetch the corresponding dream post JSON file
        const res = await fetch(`dreams_${animal}_posts.json`);
        const posts = await res.json();
        const randomPost = posts[Math.floor(Math.random() * posts.length)];
        postTitle.textContent = randomPost.title;
        postBody.textContent = randomPost.body;
        //postImages.innerHTML = "";

        // if (randomPost.images) {
        //     randomPost.images.forEach(imgPath => {
        //         const img = document.createElement("img");
        //         // The provided original code suggests paths might use backslashes, so we keep the replacement
        //         img.src = imgPath.replace(/\\/g, "/"); 
        //         postImages.appendChild(img);
        //     });
        // }

        popup.style.display = "flex";
    } catch (err) {
        console.error("Failed to load JSON:", err);
    }
}


closeBtn.onclick = () => { popup.style.display = "none"; animalDetected = null; };
window.onclick = e => { if (e.target == popup) { popup.style.display = "none"; animalDetected = null; } };
