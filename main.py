import pygame
import sys
import random
import librosa
import time
import numpy as np
from PIL import Image, ImageFilter

# ===========================
# Audio Analysis Functions
# ===========================

def load_mp3(file_path):
    """
    Loads an MP3 file and returns the audio time series and sampling rate.

    Parameters:
    - file_path (str): Path to the MP3 file.

    Returns:
    - y (numpy.ndarray): Audio time series.
    - sr (int): Sampling rate of `y`.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        print(f"Loaded '{file_path}' with sampling rate {sr} Hz")
        return y, sr
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None, None

def extract_beats(y, sr):
    """
    Extracts beat timings from the audio signal.

    Parameters:
    - y (numpy.ndarray): Audio time series.
    - sr (int): Sampling rate.

    Returns:
    - tempo (float): Estimated tempo (beats per minute).
    - beat_times (list of float): Times (in seconds) of each beat.
    """
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Ensure tempo is a float
        if isinstance(tempo, np.ndarray):
            # If tempo is an array, take the first element
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
        
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print(f"Estimated tempo: {tempo:.2f} BPM")
        print(f"Number of beats detected: {len(beat_times)}")
        return tempo, beat_times
    except Exception as e:
        print(f"Error extracting beats: {e}")
        return None, []

def generate_level(beat_times, num_columns=4, beat_skip=1):
    """
    Generates a level based on beat timings.

    Parameters:
    - beat_times (list of float): Times (in seconds) of each beat.
    - num_columns (int): Number of columns in the game grid.
    - beat_skip (int): Generate a tile every 'beat_skip' beats.

    Returns:
    - level (list of dict): Each dict contains 'time' and 'column'.
    """
    level = []
    for i in range(0, len(beat_times), beat_skip):
        beat = beat_times[i]
        # Ensure that the column is not equal to the previous column
        if i > 0:
            last_column = level[-1]['column']
            possible_columns = [c for c in range(num_columns) if c != last_column]
        else:
            possible_columns = list(range(num_columns))
        column = random.choice(possible_columns)
        level.append({'time': beat, 'column': column})
    print(f"Generated {len(level)} tiles for the level (beat_skip={beat_skip}).")
    return level

# ===========================
# Pygame Initialization
# ===========================

def initialize_pygame(screen_width=800, screen_height=600):
    """
    Initializes Pygame and sets up the game window.

    Parameters:
    - screen_width (int): Width of the game window.
    - screen_height (int): Height of the game window.

    Returns:
    - screen (pygame.Surface): The game window surface.
    - clock (pygame.time.Clock): Clock object to manage FPS.
    - BACKGROUND_SURFACE (pygame.Surface): Blurred background image.
    - TILE_COLOR (tuple): RGB color for tiles.
    - GREY (tuple): RGB color for UI elements.
    - key_sounds (dict): Dictionary mapping keys to their sound effects.
    - hit_sound (pygame.mixer.Sound): Sound effect for hits.
    - miss_sound (pygame.mixer.Sound): Sound effect for misses.
    """
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Piano Tiles Clone')

    # Define colors
    TILE_COLOR = (255, 255, 255)    # White tiles
    GREY = (100, 100, 100)

    clock = pygame.time.Clock()

    # Load and blur the background image
    try:
        bg_image = Image.open('beach.jpg').convert('RGB')  # Ensure image is in RGB
        blurred_bg = bg_image.filter(ImageFilter.GaussianBlur(radius=15))
        # Resize to fit the screen
        blurred_bg = blurred_bg.resize((screen_width, screen_height))
        # Convert to Pygame surface
        bg_bytes = blurred_bg.tobytes()
        bg_mode = blurred_bg.mode
        bg_size = blurred_bg.size
        BACKGROUND_SURFACE = pygame.image.fromstring(bg_bytes, bg_size, bg_mode)
    except Exception as e:
        print(f"Error loading background image: {e}")
        # If background image fails to load, fill with a solid color
        BACKGROUND_SURFACE = pygame.Surface((screen_width, screen_height))
        BACKGROUND_SURFACE.fill((0, 0, 0))  # Black background

    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load key sounds
    keys = ['q', 'w', 'e', 'r']
    key_sounds = {}
    for key in keys:
        try:
            key_sounds[key] = pygame.mixer.Sound(f'{key}.wav')
            print(f"Loaded sound for key '{key}'.")
        except:
            key_sounds[key] = None
            print(f"Sound for key '{key}' not found.")

    # Load feedback sounds
    try:
        hit_sound = pygame.mixer.Sound('hit.wav')
        print("Loaded hit sound.")
    except:
        hit_sound = None
        print("Hit sound not found.")

    try:
        miss_sound = pygame.mixer.Sound('miss.wav')
        print("Loaded miss sound.")
    except:
        miss_sound = None
        print("Miss sound not found.")

    return screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, key_sounds, hit_sound, miss_sound

# ===========================
# Tile Class
# ===========================

class Tile:
    def __init__(self, column, spawn_time, tile_speed, tile_width, tile_height, screen_height, num_columns):
        """
        Initializes a Tile object.

        Parameters:
        - column (int): Column index where the tile appears.
        - spawn_time (float): Time (in seconds) when the tile should appear.
        - tile_speed (float): Speed at which the tile moves down the screen (pixels per second).
        - tile_width (int): Width of each tile.
        - tile_height (int): Height of each tile.
        - screen_height (int): Height of the game screen.
        - num_columns (int): Total number of columns in the game.
        """
        self.column = column
        self.spawn_time = spawn_time
        self.tile_speed = tile_speed
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.screen_height = screen_height
        self.num_columns = num_columns
        self.x = column * tile_width
        self.y = -tile_height  # Start above the screen
        self.hit = False       # Indicates if the tile has been hit

    def update(self, delta_time):
        """
        Updates the tile's position based on its speed and the elapsed time.

        Parameters:
        - delta_time (float): Time elapsed since the last frame (in seconds).
        """
        self.y += self.tile_speed * delta_time

    def draw(self, surface, color, border_color=(0, 0, 0), border_width=2):
        """
        Draws the tile on the given surface with a border.

        Parameters:
        - surface (pygame.Surface): The surface to draw the tile on.
        - color (tuple): RGB color of the tile.
        - border_color (tuple): RGB color of the border.
        - border_width (int): Thickness of the border in pixels.
        """
        # Draw the filled rectangle (the tile)
        pygame.draw.rect(surface, color, (self.x, self.y, self.tile_width, self.tile_height))
        
        # Draw the border rectangle
        pygame.draw.rect(surface, border_color, (self.x, self.y, self.tile_width, self.tile_height), border_width)

# ===========================
# Button Class
# ===========================

class Button:
    def __init__(self, rect, color, text, font, text_color=(255, 255, 255)):
        """
        Initializes a Button object.

        Parameters:
        - rect (pygame.Rect): The rectangle defining the button's position and size.
        - color (tuple): RGB color of the button.
        - text (str): Text displayed on the button.
        - font (pygame.font.Font): Font object for rendering text.
        - text_color (tuple): RGB color of the text.
        """
        self.rect = rect
        self.color = color
        self.text = text
        self.font = font
        self.text_color = text_color

    def draw(self, surface):
        """
        Draws the button on the given surface.

        Parameters:
        - surface (pygame.Surface): The surface to draw the button on.
        """
        pygame.draw.rect(surface, self.color, self.rect)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        """
        Determines if the button is clicked based on mouse position.

        Parameters:
        - pos (tuple): Mouse position.

        Returns:
        - bool: True if clicked, False otherwise.
        """
        return self.rect.collidepoint(pos)

# ===========================
# Game Functions
# ===========================

def load_and_play_song(file_path):
    """
    Loads and plays an MP3 file using Pygame's mixer.

    Parameters:
    - file_path (str): Path to the MP3 file.
    """
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"Playing song: '{file_path}'")
    except Exception as e:
        print(f"Error playing '{file_path}': {e}")

def handle_tile_tap(event, tiles, num_columns, screen_width, screen_height, tile_height, score, animations, key_sounds, hit_sound, miss_sound):
    """
    Handles user input for tapping tiles.

    Parameters:
    - event (pygame.Event): The event object.
    - tiles (list of Tile): List of active tiles.
    - num_columns (int): Number of columns in the game grid.
    - screen_width (int): Width of the game window.
    - screen_height (int): Height of the game window.
    - tile_height (int): Height of each tile.
    - score (list): List containing a single integer representing the score.
    - animations (list): List to store animations for visual feedback.
    - key_sounds (dict): Dictionary mapping keys to their sound effects.
    - hit_sound (pygame.mixer.Sound): Sound effect for hits.
    - miss_sound (pygame.mixer.Sound): Sound effect for misses.
    """
    key_to_column = {
        pygame.K_q: 0,
        pygame.K_w: 1,
        pygame.K_e: 2,
        pygame.K_r: 3  # Assuming 4 columns
    }

    if event.key in key_to_column:
        pressed_column = key_to_column[event.key]
        key_char = pygame.key.name(event.key)
        print(f"Key pressed: {key_char}")

        # Play the corresponding key sound
        if key_sounds.get(key_char):
            key_sounds[key_char].play()

        # Define a hit window around the hit zone line
        hit_zone_y = screen_height - tile_height
        hit_zone_buffer = 300  # Increased for taller hit zone
        hit_zone_start = hit_zone_y - hit_zone_buffer
        hit_zone_end = hit_zone_y + hit_zone_buffer
        print(hit_zone_start, hit_zone_end)

        # Iterate over a copy of the tiles list to safely remove items
        for tile in tiles[:]:
            if tile.column == pressed_column and hit_zone_start <= tile.y <= hit_zone_end:
                print(f"Tile y-position: {tile.y}")
                tiles.remove(tile)
                score[0] += 1
                # Add a hit animation at the tile's position
                hit_animation = {
                    'type': 'circle',  # Existing animation type
                    'x': tile.x + tile.tile_width // 2,
                    'y': tile.y + tile.tile_height // 2,
                    'radius': 30,
                    'color': (0, 0, 255),  # Green color for hit
                    'alpha': 255
                }
                animations.append(hit_animation)
                # Play hit sound
                if hit_sound:
                    hit_sound.play()
                break  # Only allow one tile to be hit per key press
        else:
            # If no tile was hit, consider it a miss
            score[0] -= 1  # Penalize for wrong tap
            print(f"Missed tap on column {pressed_column}. Score: {score[0]}")
            # Add a miss animation at the bottom of the screen
            miss_animation = {
                'type': 'circle',  # Existing animation type
                'x': pressed_column * (screen_width // num_columns) + (screen_width // num_columns) // 2,
                'y': screen_height - tile_height,
                'radius': 30,
                'color': (255, 0, 0),  # Red color for miss
                'alpha': 255
            }
            animations.append(miss_animation)
            # Play miss sound
            if miss_sound:
                miss_sound.play()

def update_animations(animations, delta_time):
    """
    Updates and renders animations for visual feedback.

    Parameters:
    - animations (list of dict): List of animations to update.
    - delta_time (float): Time elapsed since the last frame (in seconds).
    """
    current_time = time.time()
    for animation in animations[:]:
        if animation['type'] == 'circle':
            # Existing circle animation logic
            animation['radius'] -= 200 * delta_time  # Adjust speed as needed
            animation['alpha'] -= 255 * delta_time   # Adjust fade speed

            if animation['radius'] <= 0 or animation['alpha'] <= 0:
                animations.remove(animation)
            else:
                # Ensure alpha stays within valid range
                alpha = max(0, min(255, int(animation['alpha'])))
                # Create a surface for the animation with per-pixel alpha
                surf_size = int(animation['radius'] * 2)
                if surf_size > 0:
                    surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
                    # Draw the circle at the center of the surface
                    pygame.draw.circle(surf, animation['color'], 
                                       (surf_size // 2, surf_size // 2), int(animation['radius']))
                    surf.set_alpha(alpha)
                    animation['current_surf'] = surf

        elif animation['type'] == 'milestone':
            # Milestone animation logic
            elapsed_time = current_time - animation['start_time']
            if elapsed_time >= animation['duration']:
                animations.remove(animation)
                continue

            # Fade out effect
            alpha = max(0, 255 - int((elapsed_time / animation['duration']) * 255))
            text_surface = animation['font'].render(animation['text'], True, animation['color'])
            text_surface.set_alpha(alpha)
            animation['current_surf'] = text_surface

def main_game_loop(song_path, level, screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, key_sounds, hit_sound, miss_sound, screen_width=800, screen_height=600, num_columns=4, hit_time=2.0):
    """
    Runs the main game loop.

    Parameters:
    - song_path (str): Path to the MP3 file.
    - level (list of dict): Level data containing tile timings and columns.
    - screen (pygame.Surface): The game window surface.
    - clock (pygame.time.Clock): Clock object to manage FPS.
    - BACKGROUND_SURFACE (pygame.Surface): Blurred background image.
    - TILE_COLOR (tuple): RGB color for tiles.
    - GREY (tuple): RGB color for UI elements.
    - key_sounds (dict): Dictionary mapping keys to their sound effects.
    - hit_sound (pygame.mixer.Sound): Sound effect for hits.
    - miss_sound (pygame.mixer.Sound): Sound effect for misses.
    - screen_width (int): Width of the game window.
    - screen_height (int): Height of the game window.
    - num_columns (int): Number of columns in the game grid.
    - hit_time (float): Time (in seconds) for a tile to reach the hit zone.
    """
    # Load and play the song
    load_and_play_song(song_path)

    # Tile properties
    tile_width = screen_width // num_columns
    tile_height = 100  # Height of each tile
    tile_speed = screen_height / hit_time  # Pixels per second

    # Initialize variables
    tiles = []
    animations = []  # List to store ongoing animations
    start_time = time.time()
    current_tile_index = 0
    total_tiles = len(level)

    score = [0]      # Using list to allow modification within functions
    max_misses = 5  # Game stops after max_misses
    misses = [0]     # Using list to allow modification

    # Define labels for QWERTY keys
    font_labels = pygame.font.SysFont(None, 24)
    column_labels = {
        0: 'Q',
        1: 'W',
        2: 'E',
        3: 'R'
    }

    # Define milestones
    MILESTONES = [100, 200, 300, 400, 500]
    achieved_milestones = []

    running = True
    while running:
        # Calculate elapsed time
        current_time = time.time() - start_time

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                handle_tile_tap(event, tiles, num_columns, screen_width, screen_height, tile_height, 
                               score, animations, key_sounds, hit_sound, miss_sound)

        # Generate tiles based on current time
        while current_tile_index < total_tiles and level[current_tile_index]['time'] <= current_time:
            column = level[current_tile_index]['column']
            new_tile = Tile(column, level[current_tile_index]['time'], tile_speed, tile_width, tile_height, 
                            screen_height, num_columns)
            tiles.append(new_tile)
            current_tile_index += 1

        # Update tiles
        delta_time = clock.get_time() / 1000.0  # Convert milliseconds to seconds
        for tile in tiles:
            tile.update(delta_time)

        # Detect missed tiles
        for tile in tiles[:]:
            if tile.y > screen_height:
                tiles.remove(tile)
                misses[0] += 1
                print(f"Missed tile in column {tile.column}. Misses: {misses[0]}")
                # Add a miss animation (e.g., red circle)
                miss_animation = {
                    'type': 'circle',  # Existing animation type
                    'x': tile.x + tile.tile_width // 2,
                    'y': tile.y + tile.tile_height // 2,
                    'radius': 30,
                    'color': (255, 0, 0),  # Red color for miss
                    'alpha': 255
                }
                animations.append(miss_animation)
                # Play miss sound
                if miss_sound:
                    miss_sound.play()
                # End game if max misses reached
                if misses[0] >= max_misses:
                    print("Too many misses. Game over.")
                    game_over_result = game_over_screen(screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, 
                                                        score[0], screen_width, screen_height)
                    if game_over_result == 'restart':
                        pygame.mixer.music.stop()
                        return 'restart'
                    else:
                        pygame.mixer.music.stop()
                        pygame.quit()
                        sys.exit()

        # Check for milestone achievements
        for milestone in MILESTONES:
            if score[0] >= milestone and milestone not in achieved_milestones:
                achieved_milestones.append(milestone)
                print(f"Milestone reached: {milestone} points!")
                
                # Create a milestone animation
                milestone_animation = {
                    'type': 'milestone',
                    'text': f'{milestone} Points!',
                    'x': screen_width // 2,
                    'y': screen_height // 2,
                    'font': pygame.font.SysFont(None, 72),
                    'color': (255, 215, 0),  # Gold color
                    'alpha': 255,
                    'duration': 2.0,  # Duration in seconds
                    'start_time': time.time()
                }
                animations.append(milestone_animation)

        # Update animations
        update_animations(animations, delta_time)

        # Draw everything
        screen.blit(BACKGROUND_SURFACE, (0, 0))

        # Define the hit zone line position
        hit_zone_y = screen_height - tile_height

        # Draw the hit zone area (semi-transparent rectangle)
        HIT_ZONE_AREA_HEIGHT = 80  # Increased height from 40 to 80
        HIT_ZONE_AREA_COLOR = (255, 0, 0, 100)  # Red with increased transparency
        hit_zone_surface = pygame.Surface((screen_width, HIT_ZONE_AREA_HEIGHT), pygame.SRCALPHA)
        hit_zone_surface.fill(HIT_ZONE_AREA_COLOR)
        screen.blit(hit_zone_surface, (0, hit_zone_y - HIT_ZONE_AREA_HEIGHT // 2))

        # Draw the hit zone line
        HIT_ZONE_COLOR = (255, 0, 0)  # Red color for visibility
        HIT_ZONE_THICKNESS = 4        # Increased thickness from 2 to 4 pixels
        pygame.draw.line(screen, HIT_ZONE_COLOR, (0, hit_zone_y), (screen_width, hit_zone_y), HIT_ZONE_THICKNESS)

        # **Add Vertical Separator Lines Between Columns**
        for col in range(1, num_columns):
            x = col * tile_width
            pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, screen_height), 2)  # White lines, 2 pixels thick

        # Draw tiles
        for tile in tiles:
            tile.draw(screen, TILE_COLOR)

        # Draw animations
        for animation in animations:
            if 'current_surf' in animation:
                if animation['type'] == 'circle':
                    # Center the animation at (x, y)
                    surf_rect = animation['current_surf'].get_rect(center=(animation['x'], animation['y']))
                    screen.blit(animation['current_surf'], surf_rect)
                elif animation['type'] == 'milestone':
                    # Position the milestone text
                    text_rect = animation['current_surf'].get_rect(center=(animation['x'], animation['y']))
                    screen.blit(animation['current_surf'], text_rect)

        # Display score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score[0]}", True, TILE_COLOR)
        screen.blit(score_text, (10, 10))

        # Display misses
        miss_text = font.render(f"Misses: {misses[0]}/{max_misses}", True, TILE_COLOR)
        screen.blit(miss_text, (10, 50))

        # Draw column labels
        for col in range(num_columns):
            label = column_labels.get(col, '')
            label_surf = font_labels.render(label, True, TILE_COLOR)
            label_rect = label_surf.get_rect(center=(col * tile_width + tile_width // 2, screen_height - 10))
            screen.blit(label_surf, label_rect)

        pygame.display.flip()

        # Maintain 60 FPS
        clock.tick(60)

        # Check if the song has ended
        if not pygame.mixer.music.get_busy() and len(tiles) == 0:
            print("Song ended. Game over.")
            game_over_result = game_over_screen(screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, 
                                                score[0], screen_width, screen_height)
            if game_over_result == 'restart':
                pygame.mixer.music.stop()
                return 'restart'
            else:
                pygame.mixer.music.stop()
                break

def start_screen(screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, screen_width=800, screen_height=600):
        """
        Displays the start screen with a start button.

        Parameters:
        - screen (pygame.Surface): The game window surface.
        - clock (pygame.time.Clock): Clock object to manage FPS.
        - BACKGROUND_SURFACE (pygame.Surface): Blurred background image.
        - TILE_COLOR (tuple): RGB color for tiles.
        - GREY (tuple): RGB color for UI elements.
        - screen_width (int): Width of the game window.
        - screen_height (int): Height of the game window.

        Returns:
        - bool: True if start button is clicked, False otherwise.
        """
        font_title = pygame.font.SysFont(None, 72)
        font_button = pygame.font.SysFont(None, 48)

        title_text = font_title.render("Piano Tiles Clone", True, TILE_COLOR)
        title_rect = title_text.get_rect(center=(screen_width//2, screen_height//3))

        button_width, button_height = 200, 80
        button_rect = pygame.Rect((screen_width - button_width)//2, screen_height//2, button_width, button_height)
        start_button = Button(button_rect, GREY, "Start", font_button)

        running = True
        while running:
            screen.blit(BACKGROUND_SURFACE, (0, 0))
            screen.blit(title_text, title_rect)
            start_button.draw(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_button.is_clicked(event.pos):
                        print("Start button clicked.")
                        return True

            pygame.display.flip()
            clock.tick(60)

def game_over_screen(screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, final_score, screen_width=800, screen_height=600):
        """
        Displays the game over screen with the final score and options to Restart or Quit.

        Parameters:
        - screen (pygame.Surface): The game window surface.
        - clock (pygame.time.Clock): Clock object to manage FPS.
        - BACKGROUND_SURFACE (pygame.Surface): Blurred background image.
        - TILE_COLOR (tuple): RGB color for tiles.
        - GREY (tuple): RGB color for UI elements.
        - final_score (int): The player's final score.
        - screen_width (int): Width of the game window.
        - screen_height (int): Height of the game window.

        Returns:
        - str: 'restart' if Restart is clicked, 'quit' otherwise.
        """
        font_over = pygame.font.SysFont(None, 72)
        font_text = pygame.font.SysFont(None, 48)
        font_button = pygame.font.SysFont(None, 40)

        over_text = font_over.render("Game Over", True, TILE_COLOR)
        over_rect = over_text.get_rect(center=(screen_width//2, screen_height//3))

        score_text = font_text.render(f"Final Score: {final_score}", True, TILE_COLOR)
        score_rect = score_text.get_rect(center=(screen_width//2, screen_height//2 - 50))

        # Restart Button
        button_width, button_height = 200, 60
        restart_rect = pygame.Rect((screen_width - button_width)//2, screen_height//2, button_width, button_height)
        restart_button = Button(restart_rect, GREY, "Restart", font_button)

        # Quit Button
        quit_rect = pygame.Rect((screen_width - button_width)//2, screen_height//2 + 80, button_width, button_height)
        quit_button = Button(quit_rect, GREY, "Quit", font_button)

        running = True
        while running:
            screen.blit(BACKGROUND_SURFACE, (0, 0))
            screen.blit(over_text, over_rect)
            screen.blit(score_text, score_rect)
            restart_button.draw(screen)
            quit_button.draw(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button.is_clicked(event.pos):
                        print("Restart button clicked.")
                        return 'restart'
                    elif quit_button.is_clicked(event.pos):
                        print("Quit button clicked.")
                        return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Escape key pressed. Quitting game.")
                        return 'quit'

            pygame.display.flip()
            clock.tick(60)

def run_piano_tiles_clone(song_path, beat_skip=1):
        """
        Runs the Piano Tiles clone game with the provided MP3 song.

        Parameters:
        - song_path (str): Path to the MP3 file.
        - beat_skip (int): Generate a tile every 'beat_skip' beats to reduce tile density.
        """
        while True:
            # Load and analyze the song
            y, sr = load_mp3(song_path)
            if y is None:
                return

            tempo, beat_times = extract_beats(y, sr)
            if tempo is None:
                return

            # Generate level with reduced number of tiles
            level = generate_level(beat_times, num_columns=4, beat_skip=beat_skip)

            # Initialize Pygame
            screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, key_sounds, hit_sound, miss_sound = initialize_pygame()

            # Show start screen
            if not start_screen(screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY):
                return

            # Run the main game loop
            game_over_result = main_game_loop(song_path, level, screen, clock, BACKGROUND_SURFACE, TILE_COLOR, GREY, 
                                             key_sounds, hit_sound, miss_sound)
            if game_over_result == 'restart':
                print("Restarting game...")
                pygame.mixer.music.stop()
                continue
            else:
                pygame.mixer.music.stop()
                break

        pygame.quit()
        sys.exit()

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path_to_your_song.mp3")
        sys.exit(1)

    song_path = sys.argv[1]

    run_piano_tiles_clone(song_path)
