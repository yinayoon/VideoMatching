# VideoMatching

// This code is for a video matching and was used SURF and BRISK for getting Local Features.

// I already tried to experiment that which feature detecter is better, so i gat a conclusion but this is just my opinion. i mean this conclusion could not generalize.

// My conclusion is that BRISK is fester than SURF and accuracy is also better. But SURF can get lots of Local Feature points. 

// So i would have to say that if you want to use feature detecter for a video matching you should use BRISK, but for just an image, SURF is better.

// Also i was using openCV 3.1, so when you use my code, you should build for using nonfree code by cmake. 

// i will write down some web sites for download material for processing building nonfree libraries.

//---------------------------------------------------------------------- for building nonfree library such as SURF, SIFT.... etc

https://github.com/opencv/opencv

https://github.com/opencv/opencv_contrib

cmake : https://cmake.org/download/ 

//----------------------------------------------------------------------

// So an summary of explanation how to build openCV nonfree libraries is difficult. 

// you could find methods for a installation if you are searching internet.

// the source code made by INA 
