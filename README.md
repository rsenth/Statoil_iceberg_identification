# Statoil/C-Core iceberg identification
 
Statoil, an international energy company needs to identify drifting icebergs in North Atlantic Ocean for safe navigation. They use a computer vision model built by C-CORE from satellite data to classify objects either as a ship or a drifting iceberg. 

This code uses deep learning computer vision models on the satellite data provided by Statoil/C-Core to identify icebergs. Specifically, it uses Inception-ResNet-V2  network. Final prediction was an ensemble of solutions provided by
Inception-ResNet-V2, ResNet50, Inception-V3, and VGG16 networks.
