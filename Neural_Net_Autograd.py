
import os
import numpy as np
import random
import re
import joblib
import pygame
from PIL import Image
import autograd.numpy as np
from autograd import grad

folder_path = "D:\my_python\Name_Train_Folder"
training_data=[]
with os.scandir(folder_path) as entries:
    for entry in entries:
        vector = np.zeros(10)
        image = Image.open(entry.path).convert('L')
        filename_temp = os.path.basename(entry)
        filename=re.findall('[0-9]+', filename_temp)[1]#extraction of a digit from a photo name
        vector[int(filename)]=1
        img_array = np.array(image)
        pixels = img_array.reshape(28, 28) / 255
        #shifts = [(0, 0), (0, -3), (0, 3)] # For noise right to left
        shifts = [(0, 0)]
        for shift in shifts:
            shifted_array = np.roll(pixels, shift, axis=(0, 1))
            training_data.append([shifted_array.reshape(-1), vector])
joblib.dump(training_data, 'training_data.joblib')

folder_path = "D:\my_python\Name_Test_Folder"
test_data=[]
with os.scandir(folder_path) as entries:
    for entry in entries:

        image = Image.open(entry.path).convert('L')
        filename_temp = os.path.basename(entry)
        filename=re.findall('[0-9]+', filename_temp)[1]#extraction of a digit from a photo name
        img_array = np.array(image)
        pixels = img_array.reshape(-1)/255
        test_data.append([pixels,int(filename)])
joblib.dump(test_data, 'test_data.joblib')

training_data_small = joblib.load('training_data.joblib')
test_data = joblib.load('test_data.joblib')


def f(z):#Сигмоида
    return 1.0/(1.0+np.exp(-z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for y, x in zip(sizes, sizes[1:])]

    def feedforward(self, a,biases,weights):
        for b, w in zip(biases, weights):
            a = f(np.dot(a, w.T) + b)
        return a

    def backprop(self,training_data,biases,weights):
        x,y=training_data
        output = self.feedforward(x,biases,weights)
        return np.sum((output - y) ** 2)

    def train(self, training_data,test_data, epochs, eta):
        grad_s = grad(self.backprop, argnum=(1,2))#Autograd
        for ep in range(epochs):
            for tr in training_data:
                gradient=grad_s(tr,self.biases,self.weights)
                for i in range(self.num_layers - 1):
                    self.biases[i]-= gradient[0][i]*eta
                    self.weights[i] -=  gradient[1][i]*eta
            if test_data:
                k = 0
                for photo, num in test_data:
                    tmp = np.argmax(self.feedforward(photo,self.biases,self.weights))
                    if tmp == num:
                        k = k + 1
                print("Epoch {0}: {1} / {2}".format(ep + 1, len(test_data), k))
            else:
                print("Epoch {0} complete".format(ep + 1))






photo_analyzer=Network([28*28,16,16,10])
photo_analyzer.train(training_data=training_data,test_data=test_data,epochs=10, eta=0.01)

joblib.dump(photo_analyzer, 'photo_analyzer.joblib')
photo_analyzer = joblib.load('photo_analyzer.joblib')

#Code for drawing numbers
print('0      1      2      3      4      5      6      7      8      9')
canvas = np.zeros((28, 28))
pygame.init()
clock = pygame.time.Clock()
FPS = 60
screen = pygame.display.set_mode((252, 252))
pygame.display.set_caption('Drawing')

WHITE = (255, 255, 255)
x, y = 0, 0
line_thickness = 21
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                pygame.draw.circle(screen, WHITE, event.pos, line_thickness // 2)
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
                canvas = np.zeros((252, 252))
        pygame.image.save(screen, "Test_photo.png")
        img = Image.open('Test_photo.png')
        new_img = img.resize((28, 28))
        new_img.save('Test_photo.png', 'PNG')
        image = Image.open('Test_photo.png').convert('L')
        img_array = np.array(image)
        pixels = img_array.reshape(-1) / 255
        a = photo_analyzer.feedforward(pixels)
        summa = np.sum(a)
        percent = a / summa * 100

        print(f"\r{round(percent[0], 2)}%  {round(percent[1], 2)}%  {round(percent[2], 2)}%  {round(percent[3], 2)}%  {round(percent[4], 2)}%  {round(percent[5], 2)}%  {round(percent[6], 2)}%  {round(percent[7], 2)}%  {round(percent[8], 2)}%  {round(percent[9], 2)}%",end="",flush=True)

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
