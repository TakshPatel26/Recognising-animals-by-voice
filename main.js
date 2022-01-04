function startClassification(){
  navigator.mediaDevices.getUserMedia({ audio: true});
  classifier = ml5.soundClassifier('https://teachablemachine.withgoogle.com/models/u02fOYRBx/model.json', modelReady);
}
function modelReady(){
  classifier.classify(gotResults);
}
function gotResults(error,results) {
     if (error){
       console.error(error);
     } 
     else{
     console.log(results);
     }
}