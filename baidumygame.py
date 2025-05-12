
import pygame 
import sys
import time
import random
import cv2
import mediapipe as mp 
import numpy as np

# 初始化pygame
pygame.init()

# 游戏常量
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800  # 增加高度以容纳文字区域
MAZE_SIZE = 500  # 迷宫尺寸
MAZE_ROWS = 11   # 11×11迷宫
CELL_SIZE = MAZE_SIZE // MAZE_ROWS
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 创建游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("警察抓小偷")
clock = pygame.time.Clock()

# 初始化摄像头和手势识别
def init_camera():
    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("摄像头无法打开")
        print("摄像头已成功打开")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return None

# 独立摄像头窗口显示
def show_camera_window():
    cap = init_camera()
    if cap is None:
        return
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头帧读取失败")
                break
                
            # 手势识别与状态更新
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手势骨架
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 检测并更新方向状态
                    landmarks = hand_landmarks.landmark
                    new_direction = detect_direction(landmarks)
                    if new_direction:
                        gesture_state.current_direction = new_direction
                        gesture_state.last_update = time.time()
                        cv2.putText(frame, new_direction, 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        # 1秒后清除无更新的方向状态
                        if time.time() - gesture_state.last_update > 1.0:
                            gesture_state.current_direction = None
            
            # 显示窗口
            cv2.imshow('手势识别', frame)
            
            # 检查窗口关闭事件
            if cv2.getWindowProperty('手势识别', cv2.WND_PROP_VISIBLE) < 1:
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

# 全局手势状态
class GestureState:
    def __init__(self):
        self.current_direction = None
        self.last_update = 0

gesture_state = GestureState()

# 游戏状态
class GameState:
    def __init__(self):
        self.maze = self.generate_maze()
        self.police_pos = [0, 0]  # 左上角
        self.thief_pos = [0, MAZE_ROWS-1]  # 左下角
        self.exit_pos = [MAZE_ROWS-1, 0]  # 左下角(修改后的出口位置)
        self.game_over = False
        self.thief_caught = False
        self.thief_escaped = False
        self.last_thief_move = 0
        self.thief_speed = 1.0  # 固定移动速度，每thief_speed秒移动一个格子
        self.start_time = time.time()  # 记录游戏开始时间
        self.end_time = None  # 记录游戏结束时间
        
    def generate_maze(self):
        # 生成复杂但可行的迷宫
        maze = [[1 for _ in range(MAZE_ROWS)] for _ in range(MAZE_ROWS)]
        
        # 使用深度优先搜索生成迷宫
        stack = [(0, 0)]
        maze[0][0] = 0
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx*2, y + dy*2
                if 0 <= nx < MAZE_ROWS and 0 <= ny < MAZE_ROWS and maze[nx][ny] == 1:
                    neighbors.append((nx, ny, x+dx, y+dy))
            
            if neighbors:
                nx, ny, wx, wy = random.choice(neighbors)
                maze[nx][ny] = 0
                maze[wx][wy] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # 确保终点可达
        path = find_path(maze, [0,0], [MAZE_ROWS-1, MAZE_ROWS-1])
        if not path:
            # 如果不可达，创建一条直接路径
            for i in range(MAZE_ROWS):
                maze[i][i] = 0
                if i > 0:
                    maze[i-1][i] = 0
        
        return maze

def detect_direction(landmarks):
    """增强版方向检测"""
    # 获取关键点坐标
    wrist = landmarks[0]
    index_mcp = landmarks[5]  # 食指根部
    index_tip = landmarks[8]  # 食指尖端
    thumb_tip = landmarks[4]  # 拇指尖端
    
    # 计算食指方向向量
    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y
    
    # 增加方向检测阈值
    min_movement = 0.15  # 降低灵敏度，提高最小移动阈值
    
    # 确定主要方向
    if abs(dx) > abs(dy):
        if abs(dx) > min_movement:
            return "RIGHT" if dx > 0 else "LEFT"
    elif abs(dy) > min_movement:
        return "DOWN" if dy > 0 else "UP"
    
    return None  # 未检测到明确方向

# 主游戏循环
def main():
    # 初始化摄像头和方向变量
    cap = cv2.VideoCapture(0)
    current_direction = None
    game_state = GameState()
    
    while True:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                sys.exit()
        
        # 手势识别
        if cap is None:
            # 显示摄像头未连接提示
            font = pygame.font.SysFont(None, 30)
            warning_text = font.render("摄像头未连接!", True, RED)
            screen.blit(warning_text, (SCREEN_WIDTH - 200, 20))
        else:
            ret, frame = cap.read()
            if not ret:
                print("摄像头帧读取失败")
                current_direction = None
            else:
                print("成功获取摄像头帧")  # 调试输出
                # 水平翻转摄像头画面实现镜像效果
                frame = cv2.flip(frame, 1)  # 1表示水平翻转
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                current_direction = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制手势骨架
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # 检测方向
                        landmarks = hand_landmarks.landmark
                        current_direction = detect_direction(landmarks)
            
            # 显示摄像头画面 - 调整到右侧
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)  # 旋转90度以适应显示
                frame = pygame.surfarray.make_surface(frame)
                print("摄像头画面处理成功")  # 调试输出
                
                # 强制显示摄像头画面区域
                cam_width = 300
                cam_height = 225
                cam_x = SCREEN_WIDTH - cam_width - 20
                cam_y = 20
                frame_scaled = pygame.transform.scale(frame, (cam_width, cam_height))
                screen.blit(frame_scaled, (cam_x, cam_y))
                
                # 添加摄像头标签
                font = pygame.font.SysFont(None, 24)
                speaker_text = font.render("摄像头画面", True, GREEN)
                screen.blit(speaker_text, (cam_x + cam_width//2 - 50, cam_y + cam_height + 5))
                
            except Exception as e:
                print(f"摄像头画面处理失败: {e}")
                # 显示黑色画面作为占位
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame = pygame.surfarray.make_surface(frame)
                cam_width = 300
                cam_height = 225
                cam_x = SCREEN_WIDTH - cam_width - 20
                cam_y = 20
                frame_scaled = pygame.transform.scale(frame, (cam_width, cam_height))
                screen.blit(frame_scaled, (cam_x, cam_y))
                # 显示错误信息
                font = pygame.font.SysFont(None, 24)
                error_text = font.render("摄像头不可用", True, RED)
                screen.blit(error_text, (cam_x + cam_width//2 - 50, cam_y + cam_height//2))
            
            # 绘制右侧显示区域
            right_panel_x = MAZE_SIZE + 60
            right_panel_y = 50
            right_panel_width = 400
            right_panel_height = 700
            
            # 手势识别状态显示区域(黑色背景)
            gesture_display_x = right_panel_x + 20
            gesture_display_y = right_panel_y + 20
            gesture_display_width = 360
            gesture_display_height = 300
            pygame.draw.rect(screen, BLACK, (gesture_display_x, gesture_display_y, gesture_display_width, gesture_display_height))
            
            # 显示摄像头画面(调整到右上角)
            cam_width = 300  # 画面宽度
            cam_height = 225  # 画面高度
            cam_x = SCREEN_WIDTH - cam_width - 20  # 右上角固定位置
            cam_y = 20
            
            # 缩放摄像头画面
            frame_scaled = pygame.transform.scale(frame, (cam_width, cam_height))
            screen.blit(frame_scaled, (cam_x, cam_y))
            
            # 添加摄像头标签
            font = pygame.font.SysFont(None, 24)
            speaker_text = font.render("摄像头画面", True, GREEN)
            screen.blit(speaker_text, (cam_x + cam_width//2 - 50, cam_y + cam_height + 5))
            
            # 显示方向文字(居中)
            if current_direction:
                font = pygame.font.SysFont(None, 72)
                text = font.render(current_direction, True, GREEN)
                text_rect = text.get_rect(center=(gesture_display_x + gesture_display_width//2, 
                                                gesture_display_y + gesture_display_height//2))
                screen.blit(text, text_rect)
            
            # 显示方向提示
            if current_direction:
                font = pygame.font.SysFont(None, 50)
                text = font.render(current_direction, True, BLUE)
                screen.blit(text, (MAZE_SIZE + 250, 300))
        
        # 可靠的手势控制系统
        if not game_state.game_over:
            old_pos = game_state.police_pos.copy()
            speed = 0.08  # 移动速度
            
            # 调试信息
            current_direction = gesture_state.current_direction
            if current_direction:
                print(f"有效方向指令: {current_direction} (更新时间: {time.time() - gesture_state.last_update:.1f}秒前)")
            
            # 响应手势移动
            if current_direction:
                new_pos = game_state.police_pos.copy()
                if current_direction == "UP":
                    new_pos[0] = max(0, game_state.police_pos[0] - speed)
                elif current_direction == "DOWN":
                    new_pos[0] = min(MAZE_ROWS-1, game_state.police_pos[0] + speed)
                elif current_direction == "LEFT":
                    new_pos[1] = max(0, game_state.police_pos[1] - speed)
                elif current_direction == "RIGHT":
                    new_pos[1] = min(MAZE_ROWS-1, game_state.police_pos[1] + speed)
                
                # 精确圆形碰撞检测
                cell_center_x = 50 + new_pos[1]*CELL_SIZE + CELL_SIZE//2
                cell_center_y = 50 + new_pos[0]*CELL_SIZE + CELL_SIZE//2
                radius = CELL_SIZE//2 - 3
                
                # 检查圆形边界是否碰到黑色区域
                safe_move = True
                for dx, dy in [(0,0), (radius,0), (-radius,0), (0,radius), (0,-radius)]:
                    check_x = int((cell_center_x + dx - 50) / CELL_SIZE)
                    check_y = int((cell_center_y + dy - 50) / CELL_SIZE)
                    
                    if (0 <= check_x < MAZE_ROWS and 0 <= check_y < MAZE_ROWS and 
                        game_state.maze[check_y][check_x] == 1):
                        safe_move = False
                        break
                
                if safe_move:
                    game_state.police_pos = new_pos
                    
                    # 碰撞检测
                    check_collision(game_state)
            
            # 实时显示方向状态
            font = pygame.font.SysFont(None, 30)
            status_text = [
                f"手势状态: {current_direction if current_direction else '等待输入'}",
                f"最后更新: {time.time() - gesture_state.last_update:.1f}秒前",
                "操作提示: 伸直食指明确指向方向"
            ]
            
            for i, text in enumerate(status_text):
                text_surface = font.render(text, True, BLACK)
                screen.blit(text_surface, (MAZE_SIZE + 50, 400 + i * 30))
        
        # 更新游戏状态
        update(game_state)
        
        # 绘制游戏
        draw(game_state)
        
        pygame.display.flip()
        clock.tick(FPS)

def update(game_state):
    current_time = time.time()
    if current_time - game_state.last_thief_move > 1.0:  # 每秒移动一次
        game_state.last_thief_move = current_time
        # 改进的小偷自动寻路逻辑
        if not game_state.game_over:
            # 计算到出口的路径
            path = find_path(game_state.maze, game_state.thief_pos, game_state.exit_pos)
            if path and len(path) > 1:
                next_pos = path[1]  # 下一个位置
                game_state.thief_pos = [next_pos[0], next_pos[1]]
            
            # 检查碰撞和出口条件
            check_collision(game_state)
            if game_state.thief_pos == game_state.exit_pos:
                game_state.game_over = True
                game_state.thief_escaped = True
                game_state.end_time = time.time()  # 记录游戏结束时间

# A*寻路算法
def find_path(maze, start, end):
    open_set = [tuple(start)]
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, end)}
    
    while open_set:
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
        if list(current) == end:
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        open_set.remove(current)
        
        for neighbor in get_neighbors(maze, list(current)):
            neighbor_tuple = tuple(neighbor)
            tentative_g_score = g_score[current] + 1
            if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                came_from[neighbor_tuple] = current
                g_score[neighbor_tuple] = tentative_g_score
                f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, end)
                if neighbor_tuple not in open_set:
                    open_set.append(neighbor_tuple)
    
    return None

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def distance(pos1, pos2):
    """计算两个位置间的欧几里得距离"""
    return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5

# flash_screen函数已移除

def check_collision(game_state):
    """检查警察和小偷是否相遇(基于方格位置)"""
    # 获取整数位置(即所在方格)
    police_cell = [int(game_state.police_pos[0]), int(game_state.police_pos[1])]
    thief_cell = [int(game_state.thief_pos[0]), int(game_state.thief_pos[1])]
    
    if police_cell == thief_cell:
            game_state.game_over = True
            game_state.thief_caught = True
            game_state.thief_escaped = False
            game_state.end_time = time.time()  # 记录游戏结束时间
            print("警察抓住了小偷!")

def get_neighbors(maze, pos):
    neighbors = []
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
        x = pos[0] + dx
        y = pos[1] + dy
        if 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0:
            neighbors.append([x, y])
    
    return neighbors
    

def draw(game_state):
    screen.fill(WHITE)
    
    # 绘制计时器
    if game_state.game_over and game_state.end_time:
        elapsed_time = game_state.end_time - game_state.start_time
    else:
        elapsed_time = time.time() - game_state.start_time
    timer_text = f"用时: {elapsed_time:.1f}秒"
    try:
        font = pygame.font.SysFont(["SimHei", "Microsoft YaHei", "Arial Unicode MS"], 30)
    except:
        font = pygame.font.SysFont(None, 30)
    timer_surface = font.render(timer_text, True, BLACK)
    screen.blit(timer_surface, (20, 20))  # 左上角显示
    
    # 绘制主界面 (迷宫)
    pygame.draw.rect(screen, GRAY, (50, 50, MAZE_SIZE, MAZE_SIZE))
    
    # 绘制迷宫格子
    for i in range(MAZE_ROWS):
        for j in range(MAZE_ROWS):
            if game_state.maze[i][j] == 1:  # 墙
                # 绘制立体楼房
                base_x = 50 + j*CELL_SIZE
                base_y = 50 + i*CELL_SIZE
                
                # 楼房主体
                pygame.draw.rect(screen, (50, 50, 70), (base_x, base_y, CELL_SIZE, CELL_SIZE))
                
                # 窗户网格
                for win_x in range(base_x + 5, base_x + CELL_SIZE - 5, 10):
                    for win_y in range(base_y + 5, base_y + CELL_SIZE - 5, 10):
                        if random.random() > 0.3:  # 70%概率有亮灯的窗户
                            pygame.draw.rect(screen, (200, 200, 100), (win_x, win_y, 5, 5))
                
                # 3D效果边框
                pygame.draw.line(screen, (30, 30, 50), (base_x, base_y), (base_x, base_y + CELL_SIZE), 2)
                pygame.draw.line(screen, (30, 30, 50), (base_x, base_y + CELL_SIZE), (base_x + CELL_SIZE, base_y + CELL_SIZE), 2)
                
            else:  # 路径
                pygame.draw.rect(screen, WHITE, (50 + j*CELL_SIZE, 50 + i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    # 绘制出口 - 修改为左下角
    exit_rect = pygame.Rect(
        50 + 0*CELL_SIZE + 2,
        50 + (MAZE_ROWS-1)*CELL_SIZE + 2,
        CELL_SIZE - 4, CELL_SIZE - 4
    )
    pygame.draw.rect(screen, GREEN, exit_rect)
    pygame.draw.rect(screen, BLACK, exit_rect, 2)  # 添加黑色边框
    
    # 绘制警察和小偷
    police_rect = pygame.Rect(
        50 + game_state.police_pos[1]*CELL_SIZE + 2,
        50 + game_state.police_pos[0]*CELL_SIZE + 2,
        CELL_SIZE - 4, CELL_SIZE - 4
    )
    thief_rect = pygame.Rect(
        50 + game_state.thief_pos[1]*CELL_SIZE + 2,
        50 + game_state.thief_pos[0]*CELL_SIZE + 2,
        CELL_SIZE - 4, CELL_SIZE - 4
    )
    
    # 绘制带边框的圆形
    pygame.draw.circle(screen, BLACK, police_rect.center, CELL_SIZE//2 - 1)
    pygame.draw.circle(screen, BLUE, police_rect.center, CELL_SIZE//2 - 3)
    
    pygame.draw.circle(screen, BLACK, thief_rect.center, CELL_SIZE//2 - 1)
    pygame.draw.circle(screen, RED, thief_rect.center, CELL_SIZE//2 - 3)
    
    # 添加表情符号
    font_emoji = pygame.font.SysFont("Segoe UI Emoji", CELL_SIZE//2)
    police_text = font_emoji.render("👮", True, WHITE)
    thief_text = font_emoji.render("🧑", True, WHITE)
    screen.blit(police_text, police_text.get_rect(center=police_rect.center))
    screen.blit(thief_text, thief_text.get_rect(center=thief_rect.center))
    
    # 绘制辅助界面 (说明区域) - 调整到右侧
    text_panel_x = MAZE_SIZE + 50  # 迷宫右侧
    text_panel_y = 50
    text_panel_width = 300
    text_panel_height = MAZE_SIZE
    
    pygame.draw.rect(screen, (240, 240, 240), (text_panel_x, text_panel_y, text_panel_width, text_panel_height))
    font = pygame.font.SysFont(None, 30)
    title = font.render("游戏说明", True, BLACK)
    screen.blit(title, (text_panel_x + text_panel_width//2 - 50, text_panel_y + 20))
    
    # 绘制说明文字
    try:
        # 尝试使用多种中文字体
        font_small = pygame.font.SysFont(["SimHei", "Microsoft YaHei", "Arial Unicode MS"], 20)
    except:
        # 回退到支持中文的字体
        font_small = pygame.font.Font(None, 20)
    
    instructions = [
        "警察(蓝色)需要在小偷(红色)",
        "逃跑前抓住他",
        "",
        "控制方式:",
        "伸出食指指向方向",
        "控制警察移动",
        "",
        "小偷会自动寻找",
        "出口(绿色)",
        "",
        "游戏结束条件:",
        "1. 警察抓住小偷（警察胜利）",
        "2. 小偷到达出口（小偷胜利）"
    ]
    
    for i, line in enumerate(instructions):
        try:
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (text_panel_x + 20, text_panel_y + 60 + i * 30))
        except:
            # 如果中文渲染失败，跳过该行
            pass
    
    # 绘制游戏状态
    try:
        font = pygame.font.SysFont(["SimHei", "Microsoft YaHei", "Arial Unicode MS"], 36)
        if game_state.game_over:
            if game_state.thief_caught:
                elapsed_time = game_state.end_time - game_state.start_time
                text = font.render(f"警察抓住了小偷! 用时: {elapsed_time:.1f}秒", True, RED)
            elif game_state.thief_escaped:
                elapsed_time = game_state.end_time - game_state.start_time
                text = font.render(f"小偷逃走了! 用时: {elapsed_time:.1f}秒", True, BLUE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH//2, 50))
            screen.blit(text, text_rect)
    except:
        print("游戏结束文字渲染失败")

if __name__ == "__main__":
    import threading
    
    # 启动摄像头窗口线程
    camera_thread = threading.Thread(target=show_camera_window)
    camera_thread.daemon = True
    camera_thread.start()
    
    # 运行主游戏
    main()
