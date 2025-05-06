#tictactoe


class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # AI is X, human is O
    
    def print_board(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)
    
    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False
    
    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] != ' ' and row[0] == row[1] == row[2]:
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] != ' ' and self.board[0][col] == self.board[1][col] == self.board[2][col]:
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] != ' ' and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] != ' ' and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]
        
        # Check for draw
        if all(cell != ' ' for row in self.board for cell in row):
            return 'D'
        
        return None
    
    def get_empty_cells(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def minimax(self, depth, is_max, alpha, beta):
        winner = self.check_winner()
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        elif winner == 'D':
            return 0
        
        if is_max:
            best_score = -math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'X'
                score = self.minimax(depth+1, False, alpha, beta)
                self.board[row][col] = ' '
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'O'
                score = self.minimax(depth+1, True, alpha, beta)
                self.board[row][col] = ' '
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
    
    def find_best_move(self):
        best_score = -math.inf
        best_move = None
        for row, col in self.get_empty_cells():
            self.board[row][col] = 'X'
            score = self.minimax(0, False, -math.inf, math.inf)
            self.board[row][col] = ' '
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move

# Game loop
game = TicTacToe()
while True:
    game.print_board()
    winner = game.check_winner()
    if winner:
        print("Game over!", "Draw" if winner == 'D' else f"{winner} wins!")
        break
    
    if game.current_player == 'O':
        # Human turn
        while True:
            try:
                row, col = map(int, input("Enter row and column (0-2): ").split())
                if game.make_move(row, col, 'O'):
                    break
                print("Invalid move! Try again.")
            except:
                print("Invalid input! Use format: row column (e.g., 1 2)")
        game.current_player = 'X'
    else:
        # AI turn
        print("AI is thinking...")
        row, col = game.find_best_move()
        game.make_move(row, col, 'X')
        game.current_player = 'O'
        
        
        
        
        
        
        
        
        
        
        

#coin game (two player)



class CoinGame:
    def __init__(self, coins):
        self.coins = coins
        self.max_score = 0
        self.min_score = 0
    
    def evaluate(self, coins):
        return coins[0] - coins[-1]
    
    def game_over(self, coins):
        return len(coins) == 0
    
    def get_possible_moves(self, coins):
        return ['left', 'right'] if len(coins) > 1 else ['left']
    
    def make_move(self, coins, move):
        if move == 'left':
            return coins[1:]
        return coins[:-1]

class MaxPlayer:
    def __init__(self, game):
        self.game = game
    
    def play(self, coins, depth, alpha, beta):
        if depth == 0 or self.game.game_over(coins):
            return self.game.evaluate(coins), None
        
        max_eval = -math.inf
        best_move = None
        for move in self.game.get_possible_moves(coins):
            new_coins = self.game.make_move(coins, move)
            eval, _ = MinPlayer(self.game).play(new_coins, depth-1, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move

class MinPlayer:
    def __init__(self, game):
        self.game = game
    
    def play(self, coins, depth, alpha, beta):
        if depth == 0 or self.game.game_over(coins):
            return self.game.evaluate(coins), None
        
        min_eval = math.inf
        best_move = None
        for move in self.game.get_possible_moves(coins):
            new_coins = self.game.make_move(coins, move)
            eval, _ = MaxPlayer(self.game).play(new_coins, depth-1, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Example usage with agents
coins = [3, 9, 1, 2, 7, 5]
game = CoinGame(coins)
max_player = MaxPlayer(game)
min_player = MinPlayer(game)

# Simulate a game
remaining_coins = coins.copy()
max_turn = True
while remaining_coins:
    if max_turn:
        score, move = max_player.play(remaining_coins, len(remaining_coins), -math.inf, math.inf)
        game.max_score += remaining_coins[0] if move == 'left' else remaining_coins[-1]
    else:
        score, move = min_player.play(remaining_coins, len(remaining_coins), -math.inf, math.inf)
        game.min_score += remaining_coins[0] if move == 'left' else remaining_coins[-1]
    
    print(f"{'Max' if max_turn else 'Min'} picks {'left' if move == 'left' else 'right'} coin ({remaining_coins[0] if move == 'left' else remaining_coins[-1]})")
    remaining_coins = game.make_move(remaining_coins, move)
    max_turn = not max_turn

print(f"\nFinal scores - Max: {game.max_score}, Min: {game.min_score}")
print(f"Winner: {'Max' if game.max_score > game.min_score else 'Min'}")



