# Muscle-Contraction-Analysis

This project allows researchers in electrophysiology to create traces for muscle contraction recordings. 

The code is capable of generating trace graphs based on when it detects pixel changes indicating a contraction. This code was tested using videos of animal heart tissues that were being electrically stimulated with fabricated free-standing nanoelectronic probes. This code allows the user to verify that the frequency at which the tissues were being stimulated at is indeed in line with expectations. The code plots spikes to trace the contractions as they occur and calculates the rate of contractions for the user. The code accepts any video format supported by OpenCV including those recorded by phones and other video capture devices such as AVI, MP4 and MOV. The ideal frame rate of the videos is 60 FPS but if recordings have been taken at 30 FPS then only one minor tweak in the code needs to be made. This tweak is mentioned in the code itself. 
