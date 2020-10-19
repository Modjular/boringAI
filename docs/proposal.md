---
layout: default
title:  Proposal
---

## Summary of the Project (30 points)
Our project is to train an AI that uses Computer Vision to recognize a minecraft item equivalent to any real-life image from the internet along with recognizing items in game. Based on its judgement, the agent then sets out to find the materials needed to make the item. All types of blocks and materials are scattered around the agent, so the agent would move around, recognize, and collect the material required for the recipe. 

## AI/ML Algorithms (10 points)
We plan to use Tensorflow to implement an k-nearest neighbors algorithm for image recognition, keeping in mind other algorithms available, such as AdaBoost, Random forests, etc.

## Evaluation Plan (30 points)
Quantitative
Our baseline is having our metrics be 7 of successful recognitions out of 10 test cases. We will measure % confidence over all classes we choose. We plan to start out with just 10 classes, but our hope/moonshot would be to be able to classify all items in Minecraft.
Qualitative
Our sanity case would be verifying that the standard image shown in the inventory is correctly outputted as the right class. Visual confirmation will be used that the prediction looks similar to the actual image. For example, a real life peach being classified as an apple. Our moonshot cases would be correctly verifying both in game and real life images in all angles and environments.


## Appointment with the Instructor (15 points)
Time: October 22, 9:30 AM
