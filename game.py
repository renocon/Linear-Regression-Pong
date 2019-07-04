import pygame
import sys
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


SCREEN_SIZE = WIDTH, HEIGHT = (800, 640)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
SCORE_LIMIT = 5

pygame.init()
font = pygame.font.SysFont('arial', 100)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Pong')
clock = pygame.time.Clock()
paused = False
game_over = False

source_y = -1
source_direction = -1
source_speed = 0
destination_y = -1
target = -1
df = pd.DataFrame
model = None
scaler = None
move_range = 5
mod_num = 1
speed_factor = -1.2
x_range = WIDTH // 4
safe_watch = HEIGHT // 6

# Store player and ball data in dicts
ball = {
        'x': 0,
        'y': 0,
        'colour': GREEN,
        'radius': 10,
        'velocity': {
            'x': 5,
            'y': 5
            },
        'original_velocity': {
            'x': 5,
            'y': 5
            }
        }

player = {
        'x': 0,
        'y': 0,
        'colour': WHITE,
        'width': 10,
        'height': 80,
        'velocity': {
            'x': 6,
            'y': 8
            },
        'score': {
            'value': 0,
            'x': 80,
            'y': 30
            }
        }

opponent = {
        'x': 0,
        'y': 0,
        'target' : -1,
        'colour': WHITE,
        'width': 10,
        'height': 80,
        'velocity': {
            'x': 4,
            'y': 8
            },
        'score': {
            'value': 0,
            'x': WIDTH - 150,
            'y': 30
            }
        }
def init_opponent_intelligence():
    global df
    global model
    global scaler
    df = pd.read_csv('data.csv')
    scaler = StandardScaler().fit(df)
    scaled = scaler.transform(df)
    scaledf = pd.DataFrame(scaled,columns=df.columns)
    #print(scaledf)
    model = LinearRegression().fit(scaledf[['source','direction','speed']],scaledf[['destination']])
    

def predict():
    pdf = pd.DataFrame([[source_y,source_direction,source_speed,0]],columns = df.columns)
    #predictors = np.array([[source_y,destination_y,0]])
    predictors = scaler.transform(pdf)
    predictors = pd.DataFrame(predictors, columns = df.columns)
    response = model.predict(predictors[['source','direction','speed']])
    response = scaler.inverse_transform([[0,0,0,response]])[0]
    return response[3]

def reset_characteristics():
    global source_direction
    global source_y
    global source_speed
    global destination_y
    global target
    source_y = -1
    source_direction = -1
    source_speed = 0
    destination_y = -1
    target = -1

def set_source():
    global source_direction
    global source_speed
    global source_y
    source_y = ball['y']
    source_direction = np.sign(ball['velocity']['y'])
    source_speed = ball['velocity']['x']

def set_destination():
    global destination_y
    destination_y = ball['y']

def save_record():
    global source_direction
    global source_y
    global source_speed
    global destination_y
    global df

    if source_y < 0:
        return

    row = {
        'source'        :   source_y,
        'direction'     :   source_direction,
        'speed'         :   source_speed,
        'destination'   :   destination_y
        }

    df = df.append(row, ignore_index=True)
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df.to_csv('data.csv', index=False)
    #print(df)

def ball_init():
    ball['x'] = (WIDTH // 2) - ball['radius']
    ball['y'] = random.randrange(ball['radius'] + 10, HEIGHT - ball['radius'] - 10) #(HEIGHT // 2) - ball['radius'] + 
    ball['velocity']['x'] = ball['original_velocity']['x'] * -1 # random.choice([-1, 1])
    ball['velocity']['y'] =  ball['velocity']['y'] * random.choice([-1, 1])
    pygame.time.wait(100)


def player_init():
    player['x'] = 10
    player['y'] = (HEIGHT // 2) - (player['height'] // 2)


def opponent_init():
    global target
    target = HEIGHT // 2
    opponent['x'] = WIDTH - 10 - player['width']
    opponent['y'] = (HEIGHT // 2) - (player['height'] // 2)


def init():
    reset_characteristics()
    ball_init()
    player_init()
    opponent_init()


def within_y_range(paddle, ball):
    return paddle['y'] <= ball['y'] <= paddle['y'] + paddle['height']


def ball_update():
    # Check collision first
    # Hit a paddle?
    if ((ball['x'] - ball['radius'] <= player['x'] + player['width'] 
            and within_y_range(player, ball))
            or (ball['x'] + ball['radius'] >= opponent['x'] 
                and within_y_range(opponent, ball))):
        ball['velocity']['x'] *= speed_factor
        ball['velocity']['y'] *= random.choice([-1, 1])

        if ball['x'] - ball['radius'] <= player['x'] + player['width']:
            set_source()
        if ball['x'] + ball['radius'] >= opponent['x']:
            set_destination()
            save_record()
            reset_characteristics()
        

    # Hit the top or bottom?
    if ball['y'] - ball['radius'] <= 0 or ball['y'] + ball['radius'] >= HEIGHT:
        ball['velocity']['y'] *= -1
    ball['x'] += int(ball['velocity']['x'])
    ball['y'] += int(ball['velocity']['y'])

    # Otherwise check for scoring
    if ball['x'] - ball['radius'] <= player['x']:
        opponent['score']['value'] += 1
        ball_init()
    elif ball['x'] + ball['radius'] >= opponent['x'] + opponent['width']:
        player['score']['value'] += 1
        set_destination()
        save_record()
        reset_characteristics()
        ball_init()


def player_update(key_presses):
    if player['y'] >= 0:
        if key_presses[pygame.K_w] or key_presses[pygame.K_UP]:
            player['y'] -= player['velocity']['y']
    if player['y'] + player['height'] <= HEIGHT:
        if key_presses[pygame.K_s] or key_presses[pygame.K_DOWN]:
            player['y'] += player['velocity']['y']


def opponent_update():
    global target
    new_target = 0
    ball_near = (opponent['x'] - x_range) < ball['x']
    opponent_mid = opponent['y'] + (opponent['height']/2)
    ball_mid =  ball['y'] + ball['radius']
    ball_ascending = (ball['velocity']['y'] < 0)
    ball_over_opponent = (ball_mid < (opponent_mid - safe_watch))
    ball_under_opponent = (ball_mid > (opponent_mid + safe_watch + opponent['height']))

    if source_y < 0:
        new_target = target
    #    new_target = HEIGHT // 2
    elif ball_near:
        #if ((not ball_ascending and ball_over_opponent)
        #    or (ball_ascending and ball_under_opponent)):
        #    new_target = target
        #else:
        new_target = ball['y']
    else:
        new_target = predict()
        new_target = min(HEIGHT,new_target)
        new_target = max(0, new_target)

    #if new_target != target:
    #    print('new prediction',source_y, source_direction, source_speed, new_target)
    
    target = new_target
    

    if (opponent_mid <= target - move_range
        and (opponent['y'] + opponent['height']) < HEIGHT):
        opponent['y'] += opponent['velocity']['y']
    elif (opponent['y'] > 0 
        and opponent_mid >= target + move_range): 
        opponent['y'] -= opponent['velocity']['y']

    # if ball['x'] >= WIDTH // 2:
    #     if ball['y'] < opponent['y'] and opponent['y'] >= 0:
    #         opponent['y'] -= opponent['velocity']['y']
    #     elif (ball['y'] > opponent['y'] 
    #             and (opponent['y'] + opponent['height']) <= HEIGHT):
    #         opponent['y'] += opponent['velocity']['y']
    # else:
    #     if opponent['y'] < HEIGHT // 2:
    #         opponent['y'] += opponent['velocity']['y']
    #     else:
    #         opponent['y'] -= opponent['velocity']['y']



def update(key_presses):
    ball_update()
    player_update(key_presses)
    opponent_update()


def render():
    screen.fill(BLACK)
    # Draw score
    player_score = font.render(str(player['score']['value']), True, WHITE)
    opponent_score = font.render(str(opponent['score']['value']), True, WHITE)
    screen.blit(player_score, (player['score']['x'], player['score']['y']))
    screen.blit(opponent_score, (opponent['score']['x'], opponent['score']['y']))
    # Draw ball
    pygame.draw.circle(screen, ball['colour'], (ball['x'], ball['y']),
            ball['radius'])
    # Draw players
    pygame.draw.rect(screen, player['colour'], ((player['x'], player['y']),
        (player['width'], player['height'])))
    pygame.draw.rect(screen, opponent['colour'], ((opponent['x'], opponent['y']),
        (opponent['width'], opponent['height'])))
    if game_over:
        game_over_score = font.render('Game Over', True, BLUE)
        screen.blit(game_over_score, ((WIDTH // 2) - 275, (HEIGHT // 2) - 200))
    pygame.display.update()
    clock.tick(60)


def check_game_over():
    return (player['score']['value'] == SCORE_LIMIT or
            opponent['score']['value'] == SCORE_LIMIT)


# Setup game before loop starts
init_opponent_intelligence()
init()

# Start game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                paused = not paused

    if not paused and not game_over:
        game_over = check_game_over()
        if not game_over:
            pygame.event.pump()
            update(pygame.key.get_pressed())
        render()