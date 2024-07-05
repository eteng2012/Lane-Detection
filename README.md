# Lane Detection using Classical CV and YOLOPv2 Implementations

## Description

This project contains two programs that identify road lanes on a video feed to aid in autonomous vehicle driving. These programs as well as their respective output are stored in two different folders.

The first program is a Classical CV implementation of lane detection, while the second uses YOLOPv2 to detect both lanes and vehicles on the road.

The first folder, Classical-CV-Files, contains the Classical CV implementation as well as a sample and output video file for demonstration.

The second folder, Ikomia-ML-Files, contains the YOLOPv2 implementation as well as sample image and video files for demonstration. It also has a results.json file that stores the location of the detected vehicles.

## Special Requirements

Please note that for the YOLOPv2 implementation, the Ikomia API used is not compatible with MacOS and must be run on a different system (like Google Colab). 

## Reflections

This project was a great introduction to computer vision and taught me a lot about the pros and cons of using machine learning models. 

While the classical CV program I created is a fairly lightweight and simple algorithm, it still holds some advantages over a complex machine learning model like YOLOPv2. The processing time for each frame on a given video feed is much faster due to the fairly simple mathematical tools that the classical CV algorithm uses. Furthermore, it is easy to modify and fine-tune based on the given input. For example, if the detected lines are unstable, the threshold for edge detection could be increased to reduce noise. If lines are not being detected, said thresholds could be decreased. The YOLOPv2 model, being trained on the comprehensive BDD100K dataset, has a lot of adaptability when it comes to detecting lanes. However, specific changes can not be made to the model without retraining, which is a time and resource-intensive process.

However, even with these caveats, the YOLOPv2 model performed far better on the given inputs. Real-world environments have lots of color gradients, which the classical CV lane detection process may mistakenly identify as lane markings. Even with image preprocessing techniques like color filtering and Gaussian blur, this noise impacts the stability of the detected lines and leads to lots of jittering. Furthermore, shadows and faint lane lines have a disproportional negative impact on this classical CV algorithm, as this approach uses color contrast as the primary method of detecting lanes. When there is little color contrast, it is almost impossible for this classical CV algorithm to be accurate. Convolutional Neural Networks, including YOLOPv2, have kernels that allow for more intelligent detection of unclear lanes. This allows for a certain amount of adaptability with messy real-world data that the classical CV algorithm does not have.

## Support

If you have any questions, please feel free to email me at eteng2012@gmail.com.

Thanks for stopping by!
