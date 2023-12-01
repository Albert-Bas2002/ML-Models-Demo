import cv2
import numpy as np
from PIL import Image, ImageFilter,ImageEnhance
import os
import shutil
import pygame
import tensorflow as tf
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
tf.get_logger().setLevel('ERROR')

my_class=['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и','к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

def Net_code():
    # Dataset for russian letters
    x = joblib.load('x.joblib')
    y = joblib.load('y.joblib')
    model = keras.Sequential()
    model.add(layers.Conv2D(44, (5, 5), input_shape=(238, 238,1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(31, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=3, batch_size=32)
    model.save('model.keras')

def data_get():
    directory_path = 'train'
    x=[]
    y=[]
    num_class=0
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)

        if os.path.isdir(folder_path):
            num_class=my_class.index(folder_name.lower())
            for image_filename in os.listdir(folder_path):
                if image_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_path = os.path.join(folder_path, image_filename)
                    img =Image.open(image_path)
                    width, height = img.size
                    new_width = width - 40
                    new_height = height - 40
                    left = 20
                    top = 20
                    right = new_width + 20
                    bottom = new_height + 20
                    img = img.crop((left, top, right, bottom))
                    imgs=[]
                    imgs.append(img)
                    imgs.append(img.rotate(10))
                    imgs.append(img.rotate(-10))
                    for image in imgs:
                        alpha_channel = image.split()[3]
                        img_array = np.array(alpha_channel, dtype=np.float32)
                        p= img_array / 255
                        x.append(p)
                        zero_list=np.zeros(31)
                        zero_list[num_class]=1
                        y.append(zero_list)

    x = np.array(x)
    x=np.expand_dims(x, axis=-1)
    y = np.array(y)
    joblib.dump(x, 'x.joblib')
    joblib.dump(y, 'y.joblib')


def split_list_by_y_difference(data, max_y_diff=60):
    sorted_data = sorted(data, key=lambda item: item[1])
    result = []
    current_group = []

    if sorted_data:
        current_group.append(sorted_data[0])

    for i in range(1, len(sorted_data)):
        y1 = sorted_data[i - 1][1]
        y2 = sorted_data[i][1]

        if abs(y2 - y1) > max_y_diff:
            result.append(current_group)
            current_group = []

        current_group.append(sorted_data[i])

    if current_group:
        result.append(current_group)

    return result

def pygame_dispay_text():
    canvas = np.zeros((100, 100))
    pygame.init()
    clock = pygame.time.Clock()
    FPS = 60
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption('Рисовалка')

    WHITE = (255, 255, 255)
    x, y = 0, 0
    line_thickness = 2
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:
                    pygame.draw.circle(screen, WHITE, event.pos, line_thickness //3)
                    x, y = event.pos
                    canvas[y // 10, x // 10] = 1
            elif event.type == pygame.MOUSEMOTION:

                if pygame.mouse.get_pressed()[0]:
                    pygame.draw.line(screen, WHITE, (x, y), event.pos, line_thickness)
                    x, y = event.pos
                    canvas[y // 10, x // 10] = 1
            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_d:
                    screen.fill((0, 0, 0))
                    canvas = np.zeros((900, 600))

                if event.key == pygame.K_s:
                    pygame.image.save(screen, "Test_photo_pygame.png")
                    image = Image.open("Test_photo_pygame.png")
                    image_original =cv2.imread("Test_photo_pygame.png")

                    pixels = image.load()
                    black_color = (0, 0, 0)
                    white_color = (255, 255, 255)
                    for i in range(image.width):
                        for j in range(image.height):
                            pixel = pixels[i, j]
                            if pixel == black_color:
                                pixels[i, j] = white_color
                            elif pixel == white_color:
                                pixels[i, j] = black_color
                    for x in range(image.width):
                        for y in range(image.height):
                            pixel = image.getpixel((x, y))
                            if pixel != (255, 255, 255):
                                image.putpixel((x, y), (0, 0, 0))
                    image.save("Test_photo.png")
                    image =cv2.imread('Test_photo.png')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                    img_erode = cv2.erode(thresh, np.ones((42, 42), np.uint8), iterations=1)

                    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    image_folder = 'images'
                    if os.path.exists(image_folder):
                        shutil.rmtree(image_folder)
                    os.makedirs(image_folder)

                    file_counter = 0
                    x_y_w_h_list = []
                    for idx, contour in enumerate(contours):
                        (x, y, w, h) = cv2.boundingRect(contour)
                        if hierarchy[0][idx][3] == 0:
                            x_y_w_h_list.append((x, y, w, h))

                    x_y_w_h_lists = split_list_by_y_difference(x_y_w_h_list)
                    for x_y_w_h_list in x_y_w_h_lists:
                        x_y_w_h_list = sorted(x_y_w_h_list, key=lambda item: item[0])
                        for x_y_w_h in x_y_w_h_list:
                            (x, y, w, h) = x_y_w_h
                            roi = image_original[y:y + h, x:x + w]
                            cv2.imwrite(f'images/img_{file_counter}.png', roi)
                            file_counter += 1
                        for x_y_w_h in x_y_w_h_list:
                            (x, y, w, h) = x_y_w_h
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Img", image)
                    cv2.waitKey(0)
                    new_width = 238
                    new_height = 238
                    image_files = [f for f in os.listdir('images') if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                    for image_file in image_files:
                        with Image.open(os.path.join('images', image_file)) as img:

                            img = img.resize((new_width, new_height))
                            img=img.filter(ImageFilter.GaussianBlur(radius=4))
                            enhancer = ImageEnhance.Contrast(img)
                            image_with_enhanced_contrast = enhancer.enhance(6)
                            enhancer = ImageEnhance.Brightness(image_with_enhanced_contrast)
                            img = enhancer.enhance(6)
                            img.save(os.path.join('images', image_file))
                    my_string = ''
                    model = keras.models.load_model('model_lab3.keras')

                    for image_file in image_files:
                        with Image.open(os.path.join('images', image_file)) as img:
                            img = img.convert('L')
                            img_array = np.array(img)/255
                            img_array = img_array.reshape((1, 238,238, 1))
                            predictions_indx =np.argmax(model.predict(img_array, verbose=0))
                            my_string = my_string+my_class[predictions_indx]
                    print(f"\r{my_string}",end="", flush=True)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
pygame_dispay_text()
