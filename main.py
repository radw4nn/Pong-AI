import pickle
import pygame
from pong_game import Game
import neat

SCREEN_WIDTH, SCREEN_HEIGHT = 500, 500

window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


class PongGame:
    def __init__(self, window, screen_width, screen_height):
        self.game = Game(window, screen_width, screen_height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def train_model(self, offspring_1, offspring_2, config):
        """
        Trains the model by taking both offspring and letting them play against one another and follows certain pre-set configurations.
        :param offspring_1:
        :param offspring_2:
        :param config:
        :return:
        """
        network_1 = neat.nn.FeedForwardNetwork.create(offspring_1, config)
        network_2 = neat.nn.FeedForwardNetwork.create(offspring_2, config)
        run_game = True

        while run_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

            output_1 = network_1.activate((self.left_paddle.y, abs(self.left_paddle.x - self.ball.x), self.ball.y))
            result_1 = output_1.index(max(output_1))

            output_2 = network_2.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
            result_2 = output_2.index(max(output_2))

            # Movement for left paddle
            if result_1 == 0:
                pass
            elif result_1 == 1:
                self.game.move_paddle(True, True)
            else:
                self.game.move_paddle(True, False)

            # Movement for right paddle
            if result_2 == 0:
                pass
            elif result_2 == 1:
                self.game.move_paddle(False, True)
            else:
                self.game.move_paddle(False, False)

            # print(output_1,output_2) # for debugging
            game_info = self.game.loop()
            # self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            # If either player misses the ball, then end.
            if (game_info.left_score >= 1 or game_info.right_score >= 1) or game_info.left_hits > 50:
                self.calculate_fitness(offspring_1, offspring_2, game_info)
                break


    def calculate_fitness(self, offspring_1, offspring_2, game_info):
        """
        Takes in both players and game info, and increments both of their fitness values by their corresponding hits.
        :param offspring_1:
        :param offspring_2:
        :param game_info:
        :return:
        """
        offspring_1.fitness += game_info.left_hits
        offspring_2.fitness += game_info.right_hits


    def test_model(self, genome, config):
        """
        Tests the already trained model.
        :param genome:
        :param config:
        :return:
        """
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        game_clock = pygame.time.Clock()
        run_game = True
        FPS = 60

        while run_game:
            game_clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_game = False
                    break
            key_presses = pygame.key.get_pressed()
            if key_presses[pygame.K_w]:
                self.game.move_paddle(True, True)
            if key_presses[pygame.K_s]:
                self.game.move_paddle(True, False)

            output = network.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
            result = output.index(max(output))  # Pick

            # Movement for left paddle
            if result == 0:
                pass
            elif result == 1:
                self.game.move_paddle(True, True)
            else:
                self.game.move_paddle(True, False)

            game_info = self.game.loop()
            # self.game.draw(False, True)
            pygame.display.update()

        pygame.quit()




def evaluate_fitness(genomes, config):
    """
    Evalutes the fitness of both players.
    :param genomes:
    :param config:
    :return:
    """
    global SCREEN_WIDTH, SCREEN_HEIGHT

    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    for i, (offspring_id_1, offspring_1) in enumerate(genomes):

        # Avoid out of bounds error
        if i == len(genomes) - 1:
            break

        offspring_1.fitness = 0
        for offspring_id_2, offspring_2 in genomes[
                                           i + 1:]:  # indexing range is [i+1:] to ensure that the same two offsprings don't play against one another
            offspring_2.fitness = 0 if offspring_2.fitness == None else offspring_2.fitness
            game = PongGame(window, SCREEN_WIDTH, SCREEN_HEIGHT)
            exit = game.train_model(offspring_1, offspring_2, config)
            if exit:
                quit(0)


def neat_life(config):
    """
    Creates the population, evaluates the fitness of offpsring and finds the best .
    :param config:
    :return:
    """
    # population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-n') # Load the nth generation from the nth checkpoint saved on the local directory
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    population.add_reporter(
        neat.Checkpointer(1))  # Saves a checkpoint at every nth generation. In this case I set it to 1

    Fittest = population.run(evaluate_fitness, 50)
    with open("best.pickle", 'wb') as file:
        pickle.dump(Fittest, file)


def test_ai(config):
    """
    Loads the most fit model - 'best.pickle', which is a neat.Population object type, and tests the model.
    """
    global SCREEN_WIDTH, SCREEN_HEIGHT

    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    with open("best.pickle", "rb") as file:
        fittest = pickle.load(file)

    game = PongGame(window, SCREEN_WIDTH, SCREEN_HEIGHT)
    game.test_model(fittest,config)


if __name__ == "__main__":
    config_file_path = r"config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file_path)

    Model = PongGame(window, SCREEN_WIDTH, SCREEN_HEIGHT)
    neat_life(config)
    # test_ai(config)
