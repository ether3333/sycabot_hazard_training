import numpy as np
import pygame


class SycaBotRenderer:
    def __init__(self, screen_width=500, screen_height=1000):
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.window = None
        self.clock = None

    def _to_screen(self, env, point):
        x, y = point
        sx = int((x - env.x_min) / (env.x_max - env.x_min) * self.screen_width)
        sy = int((env.y_max - y) / (env.y_max - env.y_min) * self.screen_height)
        return sx, sy

    def _cell_screen_size(self, env):
        cell_w = max(1, int(env.fire_cell_size / (env.x_max - env.x_min) * self.screen_width))
        cell_h = max(1, int(env.fire_cell_size / (env.y_max - env.y_min) * self.screen_height))
        return cell_w, cell_h

    def _draw_fire_cell(self, env, center):
        px, py = center
        cell_w, cell_h = self._cell_screen_size(env)
        rect = pygame.Rect(px - cell_w // 2, py - cell_h // 2, cell_w, cell_h)
        pygame.draw.rect(self.window, (185, 35, 20), rect)

        for _ in range(16):
            rx = int(np.random.randint(rect.left, rect.right + 1))
            ry = int(np.random.randint(rect.top, rect.bottom + 1))
            color = (255, 70 + int(np.random.randint(0, 40)), 0)
            radius = int(np.random.randint(1, 3))
            pygame.draw.circle(self.window, color, (rx, ry), radius)

    def _draw_star(self, center, radius, color):
        cx, cy = center
        points = []
        inner_radius = radius * 0.45
        for i in range(10):
            angle = -np.pi / 2.0 + i * np.pi / 5.0
            r = radius if i % 2 == 0 else inner_radius
            points.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
        pygame.draw.polygon(self.window, color, points)

    def _draw_triangle(self, center, size, color):
        cx, cy = center
        points = [
            (cx, cy - size),
            (cx - int(0.85 * size), cy + int(0.65 * size)),
            (cx + int(0.85 * size), cy + int(0.65 * size)),
        ]
        pygame.draw.polygon(self.window, color, points)

    def render(self, env):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((245, 245, 245))

        for gx in range(env.grid_shape[0]):
            for gy in range(env.grid_shape[1]):
                if env.fire_grid[gx, gy] <= 0:
                    continue
                center = env._grid_to_world_center(gx, gy)
                self._draw_fire_cell(env, self._to_screen(env, center))

        for obstacle in env.obstacles:
            start = self._to_screen(env, obstacle[0])
            end = self._to_screen(env, obstacle[1])
            pygame.draw.line(self.window, (20, 20, 20), start, end, 4)

        for exit_point in env.exits:
            self._draw_triangle(self._to_screen(env, exit_point), 10, (40, 180, 40))

        for i in range(env.num_tasks):
            color = (150, 40, 220)
            if env.task_status[i] == 2:
                color = (0, 200, 80)
            elif env.task_status[i] == 3:
                color = (180, 60, 60)
            self._draw_star(self._to_screen(env, env.tasks[i]), 9, color)

        for i in range(env.num_robots):
            x, y, th = env.robot_states[i]
            if env.robot_alive[i] > 0.5:
                color = (50, 50, 220)
                if env.robot_carrying[i] > 0.5:
                    color = (150, 40, 220)
            else:
                color = (90, 90, 90)

            center = self._to_screen(env, (x, y))
            pygame.draw.circle(self.window, color, center, 8)
            head = (int(center[0] + 16 * np.cos(th)), int(center[1] - 16 * np.sin(th)))
            left = (
                int(head[0] - 6 * np.cos(th - np.pi / 6.0)),
                int(head[1] + 6 * np.sin(th - np.pi / 6.0)),
            )
            right = (
                int(head[0] - 6 * np.cos(th + np.pi / 6.0)),
                int(head[1] + 6 * np.sin(th + np.pi / 6.0)),
            )
            pygame.draw.line(self.window, (220, 30, 30), center, head, 3)
            pygame.draw.line(self.window, (220, 30, 30), head, left, 3)
            pygame.draw.line(self.window, (220, 30, 30), head, right, 3)

        pygame.display.flip()
        self.clock.tick(env.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
