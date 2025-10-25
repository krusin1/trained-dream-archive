/**
 * Enhanced Gesture Detection for Dream Generation
 * ================================================
 * 
 * This enhanced version of the gesture detection system can:
 * 1. Detect combinations of two gestures
 * 2. Generate new dreams using the trained model
 * 3. Display the generated dreams with a beautiful interface
 * 4. Allow users to cycle through different gesture combinations
 */

let video;
let handPose;
let hands = [];

const animals = ["deer", "cat", "dog", "snake", "elephant", "flower", "rabbit", "horse", "fish", "bird"];
let detectedAnimals = [];
let lastDetectionTime = {};
let detectionCooldown = 2000; // 2 seconds cooldown between detections

// UI Elements
const popup = document.getElementById("popup");
const closeBtn = document.getElementById("close");
const postTitle = document.getElementById("post-title");
const postBody = document.getElementById("post-body");
const postImages = document.getElementById("post-images");
const loadingSpinner = document.getElementById("loading-spinner");
const generatedDreamIndicator = document.getElementById("generated-dream-indicator");

// Gesture combination tracking
let gestureCombination = [];
let maxCombinationLength = 2;

function preload() {
  handPose = ml5.handPose();
}

function setup() {
  background('#000032');
  const canvas = createCanvas(width, height);
  canvas.parent("sketch-canvas");

  video = createCapture(VIDEO);
  video.size(400, 300);
  
  video.elt.style.filter = 'grayscale(100%) contrast(150%)';
  video.elt.style.transform = 'scaleX(-1)';

  handPose.detectStart(video, gotHands);
}

function modelReady() {
  console.log("Enhanced HandPose model ready!");
}

function gotHands(results) {
  hands = results;
}

function draw() {
  image(video, 0, 0);
  
  // Draw gesture combination indicator
  drawGestureCombinationIndicator();

  if (hands.length > 0) {
    const hand = hands[0];
    detectGesture(hand);
  }
}

function drawGestureCombinationIndicator() {
  // Draw a visual indicator showing current gesture combination
  push();
  fill(255, 255, 255, 100);
  stroke(255, 255, 255, 200);
  strokeWeight(2);
  
  const indicatorY = height - 80;
  const indicatorX = 20;
  
  // Background for indicator
  rect(indicatorX - 10, indicatorY - 20, 200, 60, 10);
  
  // Title
  fill(255);
  textAlign(LEFT);
  textSize(14);
  text("Detected Animals:", indicatorX, indicatorY - 5);
  
  // Show detected animals
  textSize(12);
  if (gestureCombination.length === 0) {
    fill(200, 200, 200);
    text("Make gestures to detect animals...", indicatorX, indicatorY + 15);
  } else {
    fill(255, 255, 0);
    text(gestureCombination.join(" + "), indicatorX, indicatorY + 15);
  }
  
  // Show combination status
  if (gestureCombination.length === maxCombinationLength) {
    fill(0, 255, 0);
    text("✓ Ready to generate dream!", indicatorX, indicatorY + 35);
  } else if (gestureCombination.length > 0) {
    fill(255, 165, 0);
    text(`Need ${maxCombinationLength - gestureCombination.length} more animal(s)`, indicatorX, indicatorY + 35);
  }
  
  pop();
}

function detectGesture(hand) {
  let index = hand.index_finger_tip;
  let thumb = hand.thumb_tip;
  let middle = hand.middle_finger_tip;
  let ring = hand.ring_finger_tip;
  let pinky = hand.pinky_finger_tip;
  let wrist = hand.wrist;
  let knuckle = hand.index_finger_mcp;
  let knuckle3 = hand.middle_finger_mcp;
  let knuckle2 = hand.thumb_cmc;
  let indexpip = hand.index_finger_pip;
  let middlepip = hand.middle_finger_pip;
  let ringpip = hand.ring_finger_pip;
  
  // Calculate distances
  let d1 = dist(middle.x, middle.y, thumb.x, thumb.y);
  let d2 = dist(middle.x, middle.y, ring.x, ring.y);
  let d3 = dist(ring.x, ring.y, index.x, index.y);
  let d4 = dist(knuckle.x, knuckle.y, knuckle2.x, knuckle2.y);
  let d5 = dist(ring.x, ring.y, pinky.x, pinky.y);
  let d6 = dist(index.x, index.y, middle.x, middle.y);
  let d7 = dist(thumb.x, thumb.y, ringpip.x, ringpip.y);
  let d8 = dist(thumb.x, thumb.y, index.x, index.y);
  let d9 = dist(thumb.x, thumb.y, knuckle.x, knuckle.y);

  let detectedAnimal = null;

  // Snake gesture
  if (d2 < 15 && d3 < 15 && d5 < 17 && thumb.y < pinky.y) {
    detectedAnimal = "snake";
  }
  // Deer gesture
  else if (d1 < 15 && d2 < 15 && index.y < middle.y && d5 > 17) {
    detectedAnimal = "deer";
  }
  // Elephant gesture
  else if (index.y > thumb.y && d2 < 15 && wrist.y < index.y) {
    detectedAnimal = "elephant";
  }
  // Flower gesture
  else if (d1 > 22 && d2 > 22 && d3 > 22 && d4 < 42 && d5 < 22 && d6 > 16 && index.y < thumb.y) {
    detectedAnimal = "flower";
  }
  // Rabbit gesture
  else if (d6 < 15 && d5 < 15 && thumb.y > index.y && d7 < 20) {
    detectedAnimal = "rabbit";
  }
  // Cat gesture
  else if (d6 < 20 && indexpip.y < index.y && middlepip.y < middle.y && d5 < 20 && thumb.y > pinky.y) {
    detectedAnimal = "cat";
  }
  // Horse gesture
  else if (thumb.y < index.y && index.y < middle.y && middle.y < ring.y && ring.y < pinky.y && indexpip.y < middlepip.y && 
           d1 > 20 && d2 < 20 && d5 < 30 && knuckle.y < knuckle3.y && d8 > 15) {
    detectedAnimal = "horse";
  }
  // Fish gesture
  else if (d6 < 15 && index.y > middle.y && thumb.y < index.y && thumb.y < middle.y) {
    detectedAnimal = "fish";
  }
  // Dog gesture
  else if (d5 > 20 && d2 < 20 && (d9 < 20)) {
    detectedAnimal = "dog";
  }
  // Bird gesture
  else if (d8 < 15) {
    detectedAnimal = "bird";
  }

  // Process detected animal
  if (detectedAnimal) {
    addAnimalToCombination(detectedAnimal);
  }
}

function addAnimalToCombination(animal) {
  const currentTime = Date.now();
  
  // Check cooldown to prevent rapid detections
  if (lastDetectionTime[animal] && (currentTime - lastDetectionTime[animal]) < detectionCooldown) {
    return;
  }
  
  // Add animal to combination if not already present and we haven't reached max length
  if (!gestureCombination.includes(animal) && gestureCombination.length < maxCombinationLength) {
    gestureCombination.push(animal);
    lastDetectionTime[animal] = currentTime;
    
    console.log(`Detected: ${animal}. Combination: [${gestureCombination.join(', ')}]`);
    
    // If we have the required number of animals, generate a dream
    if (gestureCombination.length === maxCombinationLength) {
      generateCombinedDream();
    }
  }
}

function clearGestureCombination() {
  gestureCombination = [];
  lastDetectionTime = {};
}

function generateCombinedDream() {
  if (gestureCombination.length !== maxCombinationLength) {
    return;
  }
  
  console.log(`Generating dream for: ${gestureCombination.join(' + ')}`);
  
  // Show loading state
  showLoadingState();
  
  // Call the dream generation API
  fetch('/generate_dream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      animal1: gestureCombination[0],
      animal2: gestureCombination[1]
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      console.error('Error generating dream:', data.error);
      showErrorState(data.error);
    } else {
      console.log('Generated dream:', data);
      showGeneratedDream(data);
    }
  })
  .catch(error => {
    console.error('Error calling API:', error);
    showErrorState('Failed to generate dream. Please try again.');
  });
}

function showLoadingState() {
  postTitle.textContent = "Generating your dream...";
  postBody.innerHTML = `
    <div style="text-align: center; padding: 20px;">
      <div class="spinner"></div>
      <p>Combining ${gestureCombination.join(' and ')} into a new dream...</p>
    </div>
  `;
  postImages.innerHTML = "";
  
  if (generatedDreamIndicator) {
    generatedDreamIndicator.style.display = "block";
  }
  
  popup.style.display = "flex";
}

function showGeneratedDream(dreamData) {
  postTitle.textContent = dreamData.title;
  postBody.textContent = dreamData.body;
  postImages.innerHTML = `
    <div style="text-align: center; margin: 10px 0;">
      <span style="background: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
        ✨ Generated Dream ✨
      </span>
    </div>
  `;
  
  // Add some visual flair for generated dreams
  postBody.style.fontStyle = "italic";
  postBody.style.borderLeft = "4px solid #4CAF50";
  postBody.style.paddingLeft = "15px";
  
  popup.style.display = "flex";
  
  // Clear the gesture combination after showing the dream
  setTimeout(() => {
    clearGestureCombination();
  }, 3000);
}

function showErrorState(errorMessage) {
  postTitle.textContent = "Error";
  postBody.textContent = errorMessage;
  postImages.innerHTML = "";
  
  popup.style.display = "flex";
  
  // Clear the gesture combination after showing the error
  setTimeout(() => {
    clearGestureCombination();
  }, 2000);
}

// Event listeners
closeBtn.onclick = () => { 
  popup.style.display = "none"; 
  clearGestureCombination(); 
};

window.onclick = e => { 
  if (e.target == popup) { 
    popup.style.display = "none"; 
    clearGestureCombination(); 
  } 
};

// Add keyboard shortcuts for testing
document.addEventListener('keydown', function(event) {
  if (event.key === 'r' || event.key === 'R') {
    // Reset gesture combination
    clearGestureCombination();
    console.log('Gesture combination cleared');
  } else if (event.key === 'g' || event.key === 'G') {
    // Generate random dream (for testing)
    if (gestureCombination.length === 0) {
      generateRandomDream();
    }
  }
});

function generateRandomDream() {
  // Generate a random dream for testing purposes
  const randomAnimals = [...animals].sort(() => 0.5 - Math.random()).slice(0, 2);
  
  showLoadingState();
  
  fetch('/generate_random_dream')
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error('Error generating random dream:', data.error);
        showErrorState(data.error);
      } else {
        console.log('Generated random dream:', data);
        showGeneratedDream(data);
      }
    })
    .catch(error => {
      console.error('Error calling API:', error);
      showErrorState('Failed to generate random dream. Please try again.');
    });
}

// Initialize
console.log('Enhanced gesture detection system loaded');
console.log('Available animals:', animals);
console.log('Press R to reset gesture combination');
console.log('Press G to generate random dream');



