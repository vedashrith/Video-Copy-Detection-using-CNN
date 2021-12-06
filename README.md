# Video-Copy-Detection-using-CNN

   To protect the copyright of digital videos, video copy detection has become a hot topic in the field of digital copyright protection. Since a video sequence generally contains a large amount of data, to achieve efficient and effective copy detection, the key issue is to extract compact and discriminative video features. To this end, we propose a video copy detection scheme using spatio-temporal convolutional neural network (CNN) features. 

- First, we divide each video sequence into multiple video clips and sample the frames of each video clip.
- Second, the sampled frames of each video clip are fed into a pre-trained CNN model to generate the corresponding convolutional feature maps (CFMs). 
- Third, based on the generated CFMs, we extract the CNN features on the spatial and temporal domains of each video clip, i.e., the spatio-temporal CNN features.
- Finally, we build the inverted index file based on the extracted spatial-temporal CNN features with the BOW model to efficiently and effectively detect copy versions of the query from databases.
