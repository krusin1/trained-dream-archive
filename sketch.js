let video;
let handPose;
let hands = [];

const animals = ["deer", "cat", "dog", "snake", "elephant", "flower", "rabbit", "horse", "fish", "bird"];
let animalDetected = null;

const popup = document.getElementById("popup");
const closeBtn = document.getElementById("close");
const postTitle = document.getElementById("post-title");
const postBody = document.getElementById("post-body");
const postImages = document.getElementById("post-images");

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
  
 // video.hide();  

  handPose.detectStart(video, gotHands);

}

function modelReady() {
  console.log("HandPose model ready!");
}

function gotHands(results) {
  hands = results;
}

function draw() {
  image(video, 0, 0);

  if (hands.length > 0) {
    const hand = hands[0];
    detectGesture(hand);
  }

}

function detectGesture(hand) {
    console.log("here")
    let index = hand.index_finger_tip;
    let thumb = hand.thumb_tip;
    let middle = hand.middle_finger_tip;
    let ring = hand.ring_finger_tip;
    let pinky= hand.pinky_finger_tip;
    let wrist= hand.wrist;
    let knuckle= hand.index_finger_mcp;
    let knuckle3= hand.middle_finger_mcp;
    let knuckle2 = hand.thumb_cmc;
    let indexpip= hand.index_finger_pip;
    let middlepip= hand.middle_finger_pip;
    let ringpip = hand.ring_finger_pip;
    let d1= dist(middle.x,middle.y,thumb.x,thumb.y)
    let d2 = dist(middle.x,middle.y, ring.x, ring.y)
    
    
    let d3= dist(ring.x,ring.y,index.x,index.y);
    
    let d4=dist(knuckle.x,knuckle.y,knuckle2.x,knuckle2.y);
    
    let d5 = dist(ring.x,ring.y,pinky.x,pinky.y);

    let d6= dist(index.x,index.y,middle.x,middle.y);

    let d7 = dist(thumb.x,thumb.y, ringpip.x, ringpip.y);

    let d8= dist(thumb.x,thumb.y,index.x,index.y)

    let d9= dist(thumb.x, thumb.y, knuckle.x,knuckle.y)

  if (d2<15 && d3<15 && d5<17 && thumb.y<pinky.y) {
    showAnimal("snake");
    console.log("snake!")
  } 
  if (d1 < 15 && d2 < 15 && index.y < middle.y && d5 > 17) {
    showAnimal("deer");
  }
  if(index.y > thumb.y && d2<15 && wrist.y<index.y )
  {
    showAnimal("elephant");
  }
 
  if (
    d1 > 22 &&  d2 > 22 && d3 > 22 && d4 < 42 && d5 < 22 && d6 > 16 && index.y < thumb.y ) {
    showAnimal("flower");
  }
  if (d6 < 15 && d5 < 15 && thumb.y > index.y && d7<20) {
  showAnimal("rabbit");
}
  if (d6 < 20 && indexpip.y < index.y && middlepip.y < middle.y && d5 < 20 && thumb.y > pinky.y  ) {
    showAnimal("cat");
 }
  if (thumb.y< index.y && index.y < middle.y && middle.y < ring.y && ring.y < pinky.y && indexpip.y < middlepip.y  && 
  d1 > 20 && d2 < 20 && d5 < 30  && knuckle.y<knuckle3.y && d8>15) 
  {
    showAnimal("horse");

  }  


if (d6 < 15 && index.y > middle.y  && thumb.y < index.y && thumb.y < middle.y) {
  showAnimal("fish");
  console.log("fish!");
}

if (
  d5 > 20 &&  d2 < 20 &&  (d9 < 20)) {
  showAnimal("dog");
}

if(d8<15){
  showAnimal("bird");
}


}

async function showAnimal(animal) {
  if (animalDetected === animal) return; 
  animalDetected = animal;

  try {
    const res = await fetch(`dreams_${animal}_posts.json`);
    const posts = await res.json();
    const randomPost = posts[Math.floor(Math.random() * posts.length)];
    postTitle.textContent = randomPost.title;
    postBody.textContent = randomPost.body;
    postImages.innerHTML = "";

    if (randomPost.images) {
      randomPost.images.forEach(imgPath => {
        const img = document.createElement("img");
        img.src = imgPath.replace(/\\/g, "/");
        postImages.appendChild(img);
      });
    }

    popup.style.display = "flex";
  } catch (err) {
    console.error("Failed to load JSON:", err);
  }
}


closeBtn.onclick = () => { popup.style.display = "none"; animalDetected = null; };
window.onclick = e => { if (e.target == popup) { popup.style.display = "none"; animalDetected = null; } };
