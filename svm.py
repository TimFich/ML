import pygame
import numpy as np
from sklearn.svm import SVC

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("SVM")
clock = pygame.time.Clock()

RED = (255, 0, 0)
BLUE = (0, 0, 255)

points = []
labels = []


def draw(points, labels, classifier=None, new_point=None):
    screen.fill((255, 255, 255))
    for point, label in zip(points, labels):
        color = RED if label == 1 else BLUE
        pygame.draw.circle(screen, color, point, 5)

    if classifier:
        w = classifier.coef_[0]
        b = classifier.intercept_[0]
        x = np.linspace(0, width, 2)
        y = -(w[0] / w[1]) * x - b / w[1]
        pygame.draw.line(screen, (0, 255, 0), (x[0], y[0]), (x[1], y[1]), 2)

    if new_point:
        color = RED if new_point[1] == 1 else BLUE
        pygame.draw.circle(screen, color, new_point[0], 5)

    pygame.display.flip()


running = True
classifier = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and pygame.key.get_mods() & pygame.KMOD_LSHIFT:
                new_point = event.pos
                prediction = classifier.predict([new_point])
                new_label = prediction[0]
                points.append(new_point)
                labels.append(new_label)
                draw(points, labels, classifier, (new_point, new_label))
            elif event.button == 3:
                points.append(event.pos)
                labels.append(-1)
            elif event.button == 1:
                points.append(event.pos)
                labels.append(1)

        elif event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_SPACE or event.key == 32) and len(points) >= 2:
                classifier = SVC(kernel='linear')
                classifier.fit(points, labels)
                draw(points, labels, classifier)

    draw(points, labels, classifier)
    clock.tick(60)

pygame.quit()
