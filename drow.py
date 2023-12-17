import pygame

# Инициализация Pygame
pygame.init()

# Размеры окна
width, height = 560, 280

# Цвета
black = (0, 0, 0)
white = (255, 255, 255)

# Создаем окно
screen = pygame.display.set_mode((width, height))
screen.fill(white)
pygame.display.set_caption("Number Recognition")

# Основной цикл
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
            # Обработка движения мыши и рисования числа
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                pygame.draw.circle(screen, black, (mouse_x, mouse_y), 7)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                pygame.image.save(screen, "img/image.jpg")

                screen.fill(white)

    # Обновление экрана
    pygame.display.flip()

# Завершение программы при выходе из цикла
pygame.quit()
