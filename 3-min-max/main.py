import random
import time
import numpy as np
import matplotlib.pyplot as plt

random.seed(20)  # TODO: For final results set seed as your student's id modulo 42


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__ (OOOO000O000O00000 ):
        OOOO000O000O00000 .numbers =[]
    def act (O000000O000OO0O0O ,O0OO0O0O0O0OO0O00 ):
        if len (O0OO0O0O0O0OO0O00 )%2 ==0 :
            O00O0O0000000OO0O =sum (O0OO0O0O0O0OO0O00 [::2 ])
            O0O00O0OO00O0O0O0 =sum (O0OO0O0O0O0OO0O00 )-O00O0O0000000OO0O
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
                return O0OO0O0O0O0OO0O00 [1 :] # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
            return O0OO0O0O0O0OO0O00 [:-1 ]
        else :
            O00O0O0000000OO0O =max (sum (O0OO0O0O0O0OO0O00 [1 ::2 ]),sum (O0OO0O0O0O0OO0O00 [2 ::2 ]))
            O0O00O0OO00O0O0O0 =max (sum (O0OO0O0O0O0OO0O00 [:-1 :2 ]),sum (O0OO0O0O0O0OO0O00 [:-2 :2 ]))
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
                return O0OO0O0O0O0OO0O00 [:-1 ]
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
            return O0OO0O0O0O0OO0O00 [1 :]


class MinMaxAgent:
    def __init__(self, max_depth=15):
        self.numbers = []
        self.numbers_self = []
        self.numbers_enemy = []
        self.depth = max_depth


    def possible_moves(self, vector):
        moves = {}
        if len(vector) == 1:
            moves["start"] = vector[0]
            return moves
        moves["start"] = vector[0]
        moves["end"] = vector[-1]
        return moves


    def append_number(self, vector, bestMove, turn_self):
        if turn_self:
            if bestMove == "start":
                self.numbers_self.append(vector[0])
                newVector = vector[1:]
            elif bestMove == "end":
                self.numbers_self.append(vector[-1])
                newVector = vector[:-1]
            return newVector
        else:
            if bestMove == "start":
                self.numbers_enemy.append(vector[0])
                newVector = vector[1:]
            elif bestMove == "end":
                self.numbers_enemy.append(vector[-1])
                newVector = vector[:-1]
            return newVector


    def minimax(self, vector, depth, isMaximazing):
        if depth == 0 or len(vector) == 0:
            return sum(self.numbers_self) - sum(self.numbers_enemy)
        
        if isMaximazing:
            bestScore = -1e10
            moves = self.possible_moves(vector)
            for key, value in moves.items():
                new_vector = self.append_number(vector, key, True)
                score = self.minimax(new_vector, depth - 1, False)
                new_vector = vector
                self.numbers_self = self.numbers_self[:-1]
                if score > bestScore:
                    bestScore = score
            return bestScore
        else:
            bestScore = 1e10
            moves = self.possible_moves(vector)
            for key, value in moves.items():
                new_vector = self.append_number(vector, key, False)
                score = self.minimax(new_vector, depth - 1, True)
                new_vector = vector
                self.numbers_enemy = self.numbers_enemy[:-1]
                if score < bestScore:
                    bestScore = score
            return bestScore

    
    def act(self, vector: list):
        bestScore = -1e10
        bestMove = None
        moves = self.possible_moves(vector)
        for key, value in moves.items():
            new_vector = self.append_number(vector, key, True)
            score = self.minimax(new_vector, self.depth, False)
            new_vector = vector
            self.numbers_self = []
            self.numbers_enemy = []
            if score > bestScore:
                bestScore = score
                bestMove = key
        
        if bestMove == "start":
            self.numbers.append(vector[0])
            return vector[1:]
        elif bestMove == "end":
            self.numbers.append(vector[-1])
            return vector[:-1]


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def histogram():
    seconds = time.time()
    list_of_sum_diffrences = []
    for _ in range(500):
        vector = [random.randint(-10, 10) for _ in range(15)]
        first_agent, second_agent = MinMaxAgent(15), GreedyAgent()
        #first_agent, second_agent = MinMaxAgent(15), MinMaxAgent(15)
        run_game(vector, first_agent, second_agent)
        list_of_sum_diffrences.append(sum(first_agent.numbers))
    for _ in range(500):
        vector = [random.randint(-10, 10) for _ in range(15)]
        first_agent, second_agent = GreedyAgent(), MinMaxAgent(15)
        #first_agent, second_agent = MinMaxAgent(15), MinMaxAgent(15)
        run_game(vector, first_agent, second_agent)
        list_of_sum_diffrences.append(sum(second_agent.numbers))
    seconds2= time.time()
    duration = seconds2-seconds
    mean = np.mean(list_of_sum_diffrences)
    std_dev = np.std(list_of_sum_diffrences)
    print(f'{duration/1000}s - średni czas wykonywania jednej gry')
    print(f'{duration}s - całkowity czas wykonywania 1000 gier')
    print(f'{mean} - średnia sum punktów uzyskanych w rozgrywkach przez MinMaxAgent-a')
    print(f'{std_dev} - odchylenie standardowe sum punktów uzyskanych w rozgrywkach przez MinMaxAgent-a')

    plt.figure(figsize=(9, 6))
    plt.hist(list_of_sum_diffrences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Średnia arytmetyczna: {mean:.2f}')
    plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=1.5, label=f'+1 Odchylenie standardowe: {mean + std_dev:.2f}')
    plt.axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=1.5, label=f'-1 Odchylenie standardowe: {mean - std_dev:.2f}')
    plt.title('Histogram')
    plt.xlabel('Wartość sumy punktów uzyskanych przez MinMaxAgent-a')
    plt.ylabel('Częstość')
    plt.legend()

    plt.show()


def main():
    vector = [random.randint(-10, 10) for _ in range(14)]
    print(f"Vector: {vector}")
    first_agent, second_agent = MinMaxAgent(15), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n")


    # HISTOGRAM
    #histogram()
    pass


if __name__ == "__main__":
    main()
