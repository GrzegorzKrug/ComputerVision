# Depth map
### Small description

## Task is to find depth of the object. Using offset from two pictures taken in different places.
Goal is to estimate Z
![Formula](formula.png)

# Input pic
![Bowling](bowling_input.png)
## Depth map without smoothing
![Bowling](bowling_depth.png)

## Convolutional smoothing using mean filter
![Bowling](bowling_mean.png)
## Convolutional smoothing using median filter
![Bowling](bowling_median.png)

## Depth map with smaller window
![Bowling](bowling_small_window.png)

## Ground truth
![Truth](bowling_truth.png)
