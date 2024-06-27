"""Script for demonstrating Neural Style Transfer.

The code is highly inspired by the Art Generation with Neural Style Transfer
programming assignment in the Deep Learning Specialization course on Coursera.
This implementation basically follows what Andrew Ng taught in the course so
if anything doesn't make sense then you know where to look first.

Example usage:
python3 main.py
Content image path: example_content_image.png
Style image path: example_style_image.png
Percent of style to transfer into content image (float between 0.0 and 1.0
exclusive, choose 0.5 as default): 0.5
"""

import tensorflow as tf
import cv2
import numpy as np

IMAGE_SIZE = 400  # Width and height of generated image.
LEARNING_RATE = 0.01
NUM_EPOCHS = 500
CONTENT_LAYER_NAME = "block5_conv4"
STYLE_LAYER_NAMES = ["block1_conv1",
                     "block2_conv1",
                     "block3_conv1",
                     "block4_conv1",
                     "block5_conv1"]

def main():
    content_filename, style_filename, percent_transfer = get_user_input()
    content_image = load_image(content_filename)
    style_image = load_image(style_filename)
    content_image = reshape_image(content_image)
    style_image = reshape_image(style_image)
    vgg = get_vgg()
    vgg = modify_vgg(vgg, [CONTENT_LAYER_NAME] + STYLE_LAYER_NAMES)
    content_outputs = vgg(content_image)  # Forward pass content image into vgg network.
    style_outputs = vgg(style_image)  # Forward pass style image into vgg network.
    generated_image = initialize_generated_image(content_image)
    start_transferring(generated_image, vgg, content_outputs, style_outputs,
                       percent_transfer, content_image, style_image)

def get_user_input():
    """Prompts the user for inputs into the program."""
    content_filename = input("Content image path: ")
    style_filename = input("Style image path: ")
    percent_transfer = input("Percent of style to transfer into content image "
              "(float between 0.0 and 1.0 exclusive, choose 0.5 as default): ")
    percent_transfer = float(percent_transfer)
    assert percent_transfer > 0 and percent_transfer < 1
    return content_filename, style_filename, percent_transfer

def load_image(filename):
    """Loads a image given the path to it and does some transformations."""
    image = cv2.imread(filename)
    if image is None:
        print("Couldn't find/load image.")
        quit()
    assert image.shape[2] == 3  # Make sure image has 3 color channels.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    return image

def reshape_image(image):
    """Makes the image IMAGE_SIZE x IMAGE_SIZE and right shape for
    forward propagation."""
    image = make_resolution(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    return image

def make_resolution(image, new_res):
    """Makes image use new resolution through resizing and cropping."""
    height, width, _ = image.shape
    if (width, height) == new_res:
        return image
    original_image = image
    target_width, target_height = new_res
    # First, make height of image = target_height.
    # Preserve aspect ratio.
    new_width = int(width * (target_height / height))
    new_height = target_height
    image = cv2.resize(image, (new_width, new_height))
    # Second, make width of image = target_width.
    if new_width > target_width:
        # Crop center target_width cols.
        crop_start = (new_width - target_width) // 2
        image = image[:, crop_start:crop_start + target_width, :]
    elif new_width < target_width:
        # Shouldn't have matched the height first, should've matched width first.
        # Make width of image = target_width from original_image.
        # Preserve aspect ratio.
        new_width = target_width
        new_height = int(height * (target_width / width))
        image = cv2.resize(original_image, (new_width, new_height))
        # Crop center target_height rows.
        assert image.shape[0] > target_height
        crop_start = (new_height - target_height) // 2
        image = image[crop_start:crop_start + target_height, :, :]
    assert image.shape[:-1] == (target_height, target_width)
    return image

def get_vgg():
    """Returns pre-trained VGG-19 convolutional neural network."""
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                      weights="imagenet")
    vgg.trainable = False
    return vgg

def modify_vgg(vgg, layers_output_wanted):
    """Modifies vgg network so that it has 1 input and multiple outputs."""
    inputs = [vgg.input]
    outputs = [vgg.get_layer(layer_name).output
               for layer_name in layers_output_wanted]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def clip_0_1(image):
    """Clips a image so that it is in 0.0 to 1.0 range."""
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

def initialize_generated_image(content_image):
    """Initializes generated_image using content_image and some noise."""
    noise = tf.random.uniform(tf.shape(content_image), -0.25, 0.25)
    generated_image = content_image + noise
    generated_image = clip_0_1(generated_image)
    generated_image = tf.Variable(generated_image)
    return generated_image

def compute_content_cost(a_C, a_G):
    """Computes content cost using activations from content image and
    activations from generated image."""
    _, n_H, n_W, n_C = a_C.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=(-1, n_W * n_W * n_C))
    a_G_unrolled = tf.reshape(a_G, shape=(-1, n_W * n_W * n_C))
    J_content = 1/(4*n_W*n_W*n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

def compute_style_cost(a_Ss, a_Gs):
    """Computes style cost using activations from style image and
    activations from generated image."""
    J_style = 0
    weight_per_layer = 1 / len(a_Ss)
    for i in range(len(a_Ss)):
        J_style += weight_per_layer * compute_layer_style_cost(a_Ss[i],
                                                               a_Gs[i])
    return J_style

def compute_layer_style_cost(a_S, a_G):
    """Computes style cost using activations from 1 layer of the
    vgg network."""
    _, n_H, n_W, n_C = a_S.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, shape=(-1, n_C)), perm=[1, 0])
    a_G = tf.transpose(tf.reshape(a_G, shape=(-1, n_C)), perm=[1, 0])
    style_matrix_S = tf.linalg.matmul(a_S, tf.transpose(a_S))
    style_matrix_G = tf.linalg.matmul(a_G, tf.transpose(a_G))
    J_style_layer = 1/(4*n_C**2*(n_H*n_W)**2) * tf.reduce_sum(tf.square(tf.subtract(style_matrix_S, style_matrix_G)))
    return J_style_layer

@tf.function()  # Optional but this decorator makes train_step run faster.
def train_step(generated_image, vgg, content_outputs, style_outputs,
               percent_transfer, optimizer):
    """Train generated_image 1 step."""
    with tf.GradientTape() as tape:
        tape.watch(generated_image)
        generated_outputs = vgg(generated_image)
        J_content = compute_content_cost(content_outputs[0],
                                         generated_outputs[0])
        J_style = compute_style_cost(style_outputs[1:], generated_outputs[1:])
        J = (1 - percent_transfer) * J_content + percent_transfer * J_style
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

def start_transferring(generated_image, vgg, content_outputs, style_outputs,
                       percent_transfer, content_image, style_image):
    """Performs neural style transfer by minimizing cost of generated_image."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    for i in range(NUM_EPOCHS):
        train_step(generated_image, vgg, content_outputs, style_outputs,
                   percent_transfer, optimizer)
        show_images(content_image, style_image, generated_image, i + 1)
        print(f"Epoch: {i + 1}")
    cv2.waitKey(0)

def show_images(image1, image2, image3, epoch_num):
    """Displays the content, style, and generated image side by side."""
    image1 = cvt_float_to_int_image(image1)
    image2 = cvt_float_to_int_image(image2)
    image3 = cvt_float_to_int_image(image3)
    combined_image = np.concatenate((image1, image2, image3), axis=1)
    cv2.destroyAllWindows()
    cv2.imshow(f"Epoch: {epoch_num}", combined_image)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        pass

def cvt_float_to_int_image(image):
    """Convert float image to numpy int image."""
    image = image[0]
    image = image * 255
    image = np.array(image).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

if __name__ == "__main__":
    main()
