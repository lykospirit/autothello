import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from ai import *

# moves: strings of length 2 (e.g 'e6', 'd4'). letters are columns, numbers are rows.
# coords: tuples of length 2 in zero-indexed row-major order (e.g (5,4), (3,3))
# indices: floats in [0,1]: (0,0) is 1/64, (7,7) is 64/64, no move is 0.
# board: an 8x8 np.array containing 0's (empty), 1's (player 1 pieces) and 2's (player 2 pieces)

def init_board():
  board = np.zeros((8,8))
  board[3][3], board[3][4], board[4][3], board[4][4] = 2, 1, 1, 2
  return board

def move_to_coord(move):
  return (ord(move[1])-49, ord(move[0])-97)

def moves_to_indices(moves):
  indices = []
  for move in moves:
    if move != '`0':
      x,y = move_to_coord(move)
      indices.append(float(x*8+y+1))
  return np.array(indices)/64.0

def coords_to_indices(coords):
  indices = []
  for x,y in coords:
    indices.append(float(x*8+y+1))
  return np.array(indices)/64.0

def indices_to_moves(indices):
  moves = []
  indices = (indices * 64.0) - 1.0
  for i,index in enumerate(indices):
    ii = int(index)
    if ii < 0: moves.append('`0')
    else: moves.append(chr(ii%8 + 97) + chr(ii//8 + 49))
  return moves

def indices_to_coords(indices):
  coords = []
  indices = (indices * 64.0) - 1.0
  for index in indices:
    ii = int(index)
    coords.append((ii//8, ii%8))
  return coords

def within_bounds(x,y):
  return x>=0 and x<8 and y>=0 and y<8

def get_valid_coords(board, player, draw=False): # returns all valid moves that player can make on board
  dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
  moves = []
  for x in range(8):
    for y in range(8):
      if board[x][y] == 0:
        for dir in dirs:
          dx, dy = x+dir[0], y+dir[1]
          nearest = None

          while within_bounds(dx,dy): # Look for flippables
            if board[dx][dy] == 0: break
            elif board[dx][dy] == player:
              nearest = (dx,dy)
              break
            dx += dir[0]
            dy += dir[1]

          if nearest is not None and (nearest[0] != x+dir[0] or nearest[1] != y+dir[1]): moves.append((x,y))

  if draw:
    draw_board = np.copy(board)
    for x,y in moves: draw_board[x][y] = 1.5
    plt.figure(figsize=(4,4))
    plt.imshow(draw_board)
    plt.show()
  return moves

def make_move(board, player, coord):
  dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
  x, y = coord
  if board[x][y] != 0:
    print("invalid move\n")
    return

  board[x][y] = player
  for dir in dirs:
    dx, dy = x+dir[0], y+dir[1]
    nearest = None

    while within_bounds(dx,dy): # Find nearest of player's pieces in dir
      if board[dx][dy] == 0: break
      elif board[dx][dy] == player:
        nearest = (dx,dy)
        break
      dx += dir[0]
      dy += dir[1]

    if nearest is not None: # If found: flip pieces in between
      dx, dy = nearest[0]-dir[0], nearest[1]-dir[1]
      while dx != x or dy != y:
        board[dx][dy] = player
        dx -= dir[0]
        dy -= dir[1]

def evaluate_board(board): # returns number of pieces: [blank, player1, player2]
  state = [0,0,0]
  for x in range(8):
    for y in range(8):
      state[int(board[x][y])] += 1
  return state

def game_execute(moves, draw=False):
  board = init_board()  # 0 - blank, 1 - black, 2 - white
  player = 1
  if draw:
    fig = plt.figure(figsize=(9,9))
    grid = ImageGrid(fig, 111, nrows_ncols=(8,8), axes_pad=0.1)

  for i,move in enumerate(moves):
    if move != '`0':
      x, y = move_to_coord(move)
      if get_valid_coords(board, player) == []:
        player = 1 if player == 2 else 2
      make_move(board, player, (x,y))

    if draw:
      if move != '`0': board[x][y] = 0.6 if player == 1 else 1.8
      grid[i].imshow(board)
      if move != '`0': board[x][y] = player
    player = 1 if player == 2 else 2

  if draw: plt.show()
  return board

def simulate_2player(AI_1, AI_2, draw=False):
  board = init_board()
  moves = []
  player = 1
  if draw:
    fig = plt.figure(figsize=(9,9))
    grid = ImageGrid(fig, 111, nrows_ncols=(8,8), axes_pad=0.1)

  while True:
    move = AI_1.move(board, moves) if player == 1 else AI_2.move(board, moves)
    x, y = move
    moves.append(move)
    make_move(board, player, move)

    if draw:
      board[x][y] = 0.6 if player == 1 else 1.8
      grid[len(moves)].imshow(board)
      board[x][y] = player

    player = 1 if player == 2 else 2
    if get_valid_coords(board, player) == []:
      player = 1 if player == 2 else 2
    if get_valid_coords(board, player) == []:
      break

  if draw: plt.show()
  return board, moves

def simulate_2player_games(games, AI_1, AI_2, draw=False):
  boards, movess = [], []
  for i in range(games):
    if i % 100 == 99: print("Running game {}...".format(i+1))
    board, moves = simulate_2player(AI_1, AI_2, draw)
    boards.append(board)
    movess.append(moves)
  return boards, movess

if __name__ == "__main__":
  # moves = np.array(['e6','f4','c3','c6','d6','c4','d3','f7','f6','f5','e3','f3','g4','h4','g5','e7','c5','b4','d7','c8','g6','e2','e8','c7','d2','f2','h3','h2','g3','h5','h6','h7','d8','f8','b5','a4','a3','c1','f1','e1','a5','b6','g7','b3','d1','h8','b1','g2','a6','c2','a2','b2','g8','g1','h1','a1','a7','b7','b8','a8'])
  # moves = np.array(['e6','f4','e3','f6','f5','d6','f3','f2','g6','g5','e2','e1','g4','h6','d1','c1','f1','g1','d3','d2','c3','h5','g3','h4','h3','h2','e7','b3','g7','e8','d8','c8','b2','b4','b5','h8','c5','c7','f8','a5','b6','a3','b1','a1','b7','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0','`0'])
  # game_execute(moves, draw=True)

  # with open("othello_database/movesets.txt", "rb") as f:
  #   outp = open("othello_database/player1.txt", "wb")
  #   for i,line in enumerate(f): # movesets.txt has 117664 lines
  #     if (i+1)%100 == 0: print("Done {}".format(i+1))
  #     moves = [i.decode('UTF-8') for i in line.strip().split()]
  #     board = game_execute(moves)
  #     state = evaluate_board(board)
  #     if state[1] > state[2]:
  #       indices = moves_to_indices(moves)
  #       outp.write(np.array2string(indices, precision=7, separator=' ', max_line_width=1000)[1:-1].encode('UTF-8'))
  #       outp.write('\n'.encode('UTF-8'))
  #   outp.close()

  filename = 'deepQ_p1_4_1000'
  AI_1 = DQN_AI(1, filename)
  # AI_1 = VAE_AI(1, filename)
  # AI_1 = AI(1)
  # filename1 = 'vae_1'
  # filename1 = 'seq2seq5'
  # filename2 = 'seq2seq4'
  # AI_1 = VAE_AI(1, filename1)
  # AI_1 = Seq2Seq_AI(1, filename1)
  # AI_1 = AI(1)
  # AI_2 = Seq2Seq_AI(2, filename2)
  AI_2 = AI(2)
  boards, movess = simulate_2player_games(500, AI_1, AI_2, draw=False)

  wins, losses, draws = 0, 0, 0
  for board in boards:
    state = evaluate_board(board)
    if state[1] > state[2]: wins += 1
    elif state[1] < state[2]: losses += 1
    else: draws += 1
  print("Results for model {}".format(filename))
  print("Player 1 - Wins: {}, Losses: {}, Draws: {}".format(wins, losses, draws))
