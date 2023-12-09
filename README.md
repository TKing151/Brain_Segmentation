# Brain_Segmentation

### Small project analyzing CNN for Tumor segmentation

## The Process:

### Step 1: Setup
#### The first code snippet is the common setup for a machine learning project using TensorFlow and Keras for image segmentation. It includes essential libraries- pandas for data manipulation, NumPy for numerical operations, seaborn and matplotlib for visualization, OpenCV for computer vision tasks, and TensorFlow for building and training neural networks. Additionally, the code sets up the environment for working with image data, including loading images, resizing, and preprocessing. The neural network architecture is based on U-Net, a popular model for image segmentation tasks. The code also imports various tools for training, early stopping, and model checkpointing.

### Step 2: The Data
#### Code snippet 2 sets up file paths for the Brain MRI segmentation dataset from the cancer imaging archive stored in the Kaggle environment (available here: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data). It then iterates through the directory structure of the specified DataPath, identifies images and corresponding masks based on filenames containing the string 'mask', and creates lists (dirs, images, and masks) to store information about the directory structure, image filenames, and mask filenames, respectively. 

### Step 3: 'Sneak peek' at data
#### This 3rd code snippet prints the first 10 entries of the masks and images lists that were created in the previous code fragment. These contain filenames of mask images (segmentation masks) and their corresponding original images, respectively. The printed output provides a glimpse of the filenames, helping to verify and understand the structure of the dataset. The next (fourth) code calculates and prints the lengths of the dirs, images, and masks lists. These lengths represent the number of directories, original images, and mask images in the dataset. It can be useful to check the size and consistency of the dataset, ensuring that the number of images and masks align correctly. The next 2 snippets (5 and 6) create a Pandas DataFrame named imagePath_df from the previously defined lists (dirs, images, and masks), then prints the first few rows of the imagePath_df DataFrame, allowing you to inspect the structure and content of the DataFrame.

### Step 4: Image shape
#### The next 2 snippets construct file paths for a randomly chosen image and its corresponding mask based on the information in the DataFrame. Using OpenCV, it reads the image and mask, and finally, it prints the shapes of the image and mask arrays. The function is designed to provide information about the dimensions of the images and masks in the dataset, aiding in understanding the data and potentially identifying any inconsistencies in sizes.

### Step 5: Plot images
#### Next 2 snippets construct file paths for a randomly chosen image and its corresponding mask based on the information in the DataFrame. Using OpenCV, it reads the image and mask and then creates a plot with three subplots: the original brain MRI image,  the segmentation mask associated with the image, and a third subplot that overlays the original MRI image with the segmentation mask, providing a visualization of the regions marked by the mask

### Step 6: 
#### 
