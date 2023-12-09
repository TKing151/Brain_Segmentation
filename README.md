# Brain_Segmentation

### Small project analyzing U-NET CNN for semantic Tumor segmentation
#### The paper 'U-Net: Convolutional Networks for Biomedical Image Segmentation' was first introduced in 2015 by by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-NET is a fast and accurate tool for semantic segmentation tasks. 

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

### Step 6: config
#### this code adds two new columns, 'image-path' and 'mask-path', to the imagePath_df DataFrame. These columns contain the complete file paths for the original images and their corresponding mask images, respectively. The paths are constructed by concatenating the DataPath, 'directory', and 'images' or 'masks' columns for each row. This step is performed to have convenient access to the full file paths when working with the images and masks in subsequent analysis or training processes.

### Step 7: Train/Test Split
#### This code uses the train_test_split function from scikit-learn to split the imagePath_df DataFrame into two subsets: train and test. The test_size parameter specifies the proportion of the dataset to include in the test split (here, 25%), and random_state ensures reproducibility by fixing the random seed.

### Step 8: hyper-parameters
#### These variables define hyperparameters for training a neural network: EPOCHS: The number of times the entire dataset is passed forward and backward through the neural network during training. BATCH_SIZE: The number of training examples utilized in one iteration. ImgHeight and ImgWidth: The height and width of the input images. Images will likely be resized to these dimensions during preprocessing. Channels: The number of color channels in the images. This is set to 3, indicating RGB color images.

### Step 9: Augmentation
#### Data augmentation is a technique used during training to artificially increase the diversity of the training dataset by applying various transformations to the existing images. This helps improve the generalization ability of the model.

### Step 10: image generator
#### This code sets up image and mask generators for training a neural network. It uses the ImageDataGenerator class from Keras to perform real-time data augmentation on the training images and masks. The parameters for data augmentation are provided by the data_augmentation dictionary defined earlier. The flow_from_dataframe method is used to generate batches of augmented images and masks from the training dataset (train). The generators are configured to rescale pixel values to the range [0, 1] (rescale=1./255.), and the target size is set to (ImgHeight, ImgWidth).

### Step 11: Validation
#### This code sets up image and mask generators for the validation dataset (test). Similar to the training generators, it uses the ImageDataGenerator class from Keras. However, in this case, data augmentation is not applied during validation. The generators are configured to rescale pixel values to the range [0, 1] These generators will provide input image and mask pairs during the validation phase, allowing the model to evaluate its performance on unseen data.

### Step 12: Data iterator
#### This code defines a custom data iterator function, data_iterator, which iterates over batches of images and masks from the training and validation generators (timage_generator and tmask_generator, and vimage_generator and vmask_generator). The function uses a generator pattern with a yield statement to produce pairs of image and mask batches. This custom iterator is designed to be used during model training and validation, allowing you to iterate over batches of augmented image and mask pairs conveniently. 

### Step 13: Conv2d_block
#### conv2d_block, defines a basic building block for a convolutional neural network (CNN). It consists of two convolutional layers with batch normalization and ReLU activation applied to the input tensor. The 2 layers perform a 2D convolution with n_filters filters, a specified kernel_size, and 'he_normal' kernel initialization. If batchnorm is set to True, batch normalization is applied, followed by a ReLU activation.

### Step 14: U-NET
#### get_unet function defines a U-Net architecture for image segmentation. The model comprises a contracting path (encoder) with four convolutional blocks, each followed by max pooling and dropout, progressively doubling the number of filters and reducing spatial dimensions. The central bottleneck contains a convolutional block without downsampling. The expansive path (decoder) consists of four upsampling blocks, each involving upsampling, concatenation with the corresponding contracting path output, and dropout. In each block, the number of filters is halved, and spatial dimensions are increased. The model's output layer utilizes a convolutional layer with one filter and sigmoid activation, producing binary segmentation masks. 

### Step 15:  Model Compilation and Summary
#### In this code snippet, an input tensor input_img is defined using the Keras Input function, representing images with dimensions (ImgHeight, ImgWidth, 3) (height, width, and three color channels). The U-Net model is then instantiated using the get_unet function with specified hyperparameters such as the number of filters (n_filters), dropout rate (dropout), and batch normalization (batchnorm). The model is compiled with the Adam optimizer, binary crossentropy loss function, and accuracy metric. Finally, a summary of the model architecture is printed, providing a concise overview of the model's layers, parameters, and output shapes. 

### Step 16: Callbacks
#### U-Net model for image segmentation is trained with specific hyperparameters and callbacks. The model is configured with an input tensor representing images of dimensions (ImgHeight, ImgWidth, 3). The get_unet function is employed to define the model architecture, featuring a contracting path, a bottleneck, and an expansive path. The model is compiled with the Adam optimizer, binary crossentropy loss function, and accuracy metric. To monitor and optimize the training process, three callbacks are implemented: EarlyStopping, which halts training if the validation loss doesn't improve for a specified number of epochs; ReduceLROnPlateau, which reduces the learning rate if validation loss stagnates; and ModelCheckpoint, which saves the model weights when there's an improvement in validation loss.

### Step 17: Batch number
#### STEP_SIZE_TRAIN and STEP_SIZE_VALID are calculated to determine the number of batches (steps) in each epoch during training and validation, respectively. The values are computed by dividing the total number of training and validation samples by the batch size (BATCH_SIZE). 

### Step 18: Training
#### U-Net model is trained using the fit method. 

### Step 19: Learning curve visualization
#### This code utilizes Matplotlib and Seaborn to generate a learning curve plot based on the training results (results). The plot displays the training and validation loss over epochs, providing insights into the model's convergence and potential overfitting. The x-axis represents the number of epochs, while the y-axis represents the log loss. The greenish teal line corresponds to the training loss, the amber line represents the validation loss, and a red marker denotes the point where the validation loss is minimized, indicating the best model. This visual representation helps assess the model's performance and generalization during training

### Step 20: Load and Evaluate
#### In the next 2 code snippets, the weights of the trained U-Net model are loaded from the file 'model-brain-mri.h5' using the load_weights method. Subsequently, the evaluate method is applied to assess the model's performance on the validation dataset (valid_gen). The number of evaluation steps is specified as STEP_SIZE_VALID. The evaluation results, including the loss and accuracy metrics, are stored in the eval_results variable. This step is crucial for validating the model's performance on unseen data and obtaining quantitative measures of its effectiveness.

### Step 21: Visual eval of model performance
#### Finally, in this loop, the code iterates 10 times, randomly selecting an index from the dataset. For each iteration, it retrieves the original image and mask paths based on the randomly selected index. The image and mask are then read using OpenCV, resized to the specified dimensions (ImgHeight and ImgWidth), and normalized to values between 0 and 1. The U-Net model is used to predict the mask for the resized image, and the original image, original mask, model prediction, and binary prediction (thresholded at 0.5) are displayed in a 1x4 grid using Matplotlib. This provides a visual comparison between the original image, ground truth mask, predicted mask, and binary mask prediction for qualitative assessment of the model's performance on random samples from the dataset.

















