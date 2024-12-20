import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx

class City:
    def __init__(self, x, y,name):
        self.x = x
        self.y = y
        self.name=name

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) +"," + str(self.name) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#permet de cree  une chemin d'une manier aleatoire
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


#permet de cree la premier population
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def classement(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):

    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    # la roue de la roulette
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            #iat: Il utilise la fonction iat de pandas pour accéder à la colonne "cum_perc" de chaque individu
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# croisement ordonné pour creer une nouvelle generation
def Crossover1(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = Crossover1(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

#deux villes échangeront leurs places sur notre itinéraire.
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



def nextGeneration(currentGen, eliteSize, mutationRate):
    #nous classons les itinéraires de la génération actuelle à l'aide de rankRoutes.
    popRanked = classement(currentGen)
    #Nous déterminons ensuite nos parents potentiels en exécutant la selectionfonction,
    selectionResults = selection(popRanked, eliteSize)
    #de créer le pool d'accouplement à l'aide de la matingPoolfonction.
    #matingpool = matingPool(currentGen, selectionResults)
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(currentGen[index])

    #nous créons ensuite notre nouvelle génération à l'aide de la breedPopulationfonction,
    children = breedPopulation(matingpool, eliteSize)
    #appliquons la mutation à l'aide de la mutatePopulationfonction.
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("distance initial: " + str(1 / classement(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("distance Final : " + str(1 / classement(pop)[0][1]))
    bestRouteIndex = classement(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

cityList = []
cityList.append(City(-5.833954,35.759465,"tanger"))
cityList.append(City(-7.5898434,33.5731104,"casablanca"))
cityList.append(City(-5.0033,34.0433,"fes"))
cityList.append(City(-7.9811,31.6295,"marakkech"))
cityList.append(City(-6.8361,34.0253,"rabat"))
cityList.append(City(-5.5500,33.8833, 'meknes'))
cityList.append(City(-6.5833,34.2500, 'kenitra'))
cityList.append(City(-7.3833,33.6833, 'mohammadeia'))
cityList.append(City(-4.8300,33.8300, 'sefrou'))
cityList.append(City(-5.3667,35.5667, 'tetouan'))


res=geneticAlgorithm(population=cityList, popSize=40, eliteSize=8, mutationRate=0.01, generations=100)
print(res)

l = []
a = 0
for i in range(0, len(res) - 1):
    t = (res[i].name, res[i + 1].name)
    l.append(t)
t = (res[i + 1].name, res[0].name)
l.append(t)

G = nx.DiGraph()

# ajouter des arcs (liens) au graphe
G.add_edges_from(l)
# dessiner le graphe
nx.draw(G, with_labels=True)
# afficher le graphique
plt.show()

lons = []
lats = []
noms = []
for ville in res:
    lons.append(ville.x)
    lats.append(ville.y)
    noms.append(ville.name)

lons.append(lons[0])
lats.append(lats[0])
noms.append(noms[0])


plt.show()
