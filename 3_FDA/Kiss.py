from __future__ import division
import Hypersphere as H
import random
import copy as copy
import Function as Function
import generateModel as GM
import numpy as np
from sklearn.utils import resample
import multiprocessing
import constants

RANDOM_SEED = 42
#tf.set_random_seed(RANDOM_SEED)

class Kiss:

    GAMMA_MIN = 1 * pow(10, -3)
    GAMMA_DECRESE_STEP = 0.5
    RADIUS_RATE = 2.41421
    INIT_BIAS = 0
    
    m_dimension = None
    m_kLevelsMax = 5
    m_stopCriterion = None
    m_upperBound = None
    m_lowerBound = None
    NumberOfEvaluations = 0
    
    #m_indexFunction = 0;
    
    m_CurrentHyperSphere = None
    
    m_CurrentLevel = None
    
    m_bestSolutionCoordinates  = []
    m_bestSolutionFitness = None
    
    m_bestSolutionCoordinatesNavigation = []
    m_bestSolutionFitnessNavigation = None
    
    m_stackMemory = {}
    m_indexes = {}
    
    m_cachMemory = {}
    
    nb_usefulDic = 0
    
    m_x_train = None
    m_y_train = None
    m_x_test = None
    m_y_test = None
    
    m_solution_architecture = []
    
    def __init__(self,dimension,kLevelsMax, upperBound,lowerBound,solution_architecture,train_data,train_labels, test_data, test_labels):
    
        #Integrate the parameters
        self.m_dimension = dimension
        self.m_kLevelsMax = kLevelsMax
        self.m_upperBound = upperBound
        self.m_lowerBound = lowerBound

        self.m_solution_architecture = []
        self.m_solution_architecture = solution_architecture
        
        #Define stop criterion
        self.m_stopCriterion = 100 * self.m_dimension
        
        #Load Data for NN training
        self.m_x_train = train_data
        self.m_y_train = train_labels
        
        self.m_x_test = test_data
        self.m_y_test = test_labels

        #Define Radius
        tempRadius = (upperBound - lowerBound)
        tempRadius = tempRadius / 2
        #Define temp Center for first CurrentHyperSPhere
        tempCenter = []

        #Initialisation de la meilleure solution au centre de l'espace + un biais
        #self.m_bestSolutionCoordinates[dimension];
    
        #Initialisation de la meilleure solution au centre de l'espace + un biais
        #self.m_bestSolutionCoordinatesNavigation = [dimension];
    
        #I to be defined
        #Intitialise the center
        for i in range(0,dimension):
            tempCenter.append(lowerBound+ ((upperBound - lowerBound) /2))
            self.m_bestSolutionCoordinates.append(lowerBound+ ((upperBound - lowerBound) /2))
            self.m_bestSolutionCoordinatesNavigation.append(lowerBound+ ((upperBound - lowerBound) /2))

        #Init Fitness at the centers
        
        with multiprocessing.Manager() as manager:
                
            temp_solution_s1 = multiprocessing.Value('d')
            p1 = multiprocessing.Process(target=self.evaluationFitness_para, args=(self.m_bestSolutionCoordinates,temp_solution_s1))

            p1.start() 
            p1.join() 
                
            self.m_bestSolutionFitness = temp_solution_s1.value
            
            
            
        
        #self.m_bestSolutionFitness = self.evaluationFitness(self.m_bestSolutionCoordinates)
        self.m_bestSolutionFitnessNavigation = self.m_bestSolutionFitness

        self.m_CurrentHyperSphere = H.Hypersphere(dimension, tempCenter, tempRadius)

        self.m_CurrentHyperSphere.m_fitness = self.m_bestSolutionFitness

        self.m_CurrentLevel = 1
        
        
        
    def DecompositionHyperSphere(self, hypersphere):
        resultList = []
        taille = hypersphere.m_dimension * 2
        sphereDimensionCreated = 1;

        for i in range(1,taille+1):
            subHyperSphere = H.Hypersphere(hypersphere.m_dimension, hypersphere.m_center, hypersphere.m_radius / self.RADIUS_RATE)

            resultList.append(subHyperSphere)
            if i % 2 == 0:
                #Create the "PLUS" hypersphere
                tempDim = (hypersphere.m_center[sphereDimensionCreated - 1]) + hypersphere.m_radius - (hypersphere.m_radius / self.RADIUS_RATE)
                resultList[i-1].m_center[sphereDimensionCreated - 1] = tempDim
                sphereDimensionCreated += 1
                #print(tempDim)
            else:
                #Create the "Minus" hypersphere
                tempDim = (hypersphere.m_center[sphereDimensionCreated - 1]) - hypersphere.m_radius + (hypersphere.m_radius / self.RADIUS_RATE)
                resultList[i-1].m_center[sphereDimensionCreated - 1] = tempDim


        return resultList


    def IntensiveLocalSearch(self):
        nbEvaluationFoisInit = self.NumberOfEvaluations
        dimension = self.m_dimension
        print("ENTER ILS")
        
        # f = open("./Results_MNIST_2D.txt", "a")
        f = open(constants.output_file, "a")
        f.write(str(-1)+"\n")
        #f.write(str(self.NumberOfEvaluations)+"\n")
        f.close()
        
        
        
        s = copy.copy(self.m_CurrentHyperSphere.m_center)
        solution_s = self.evaluationFitness(s);
        #print("IntensiveLocalSearch")
        gamma = self.m_CurrentHyperSphere.m_radius;

        bestTempSolution_s = solution_s
        count = 0
        while ((gamma > self.GAMMA_MIN) and (self.NumberOfEvaluations < self.m_stopCriterion)):
            d = 0
            while(d < dimension and (self.NumberOfEvaluations < self.m_stopCriterion)):
                #print("--Intensive Boucle")
                count += 1
                #print(count),
                #print("	"),
                #print(gamma),
                #print(" "),
                #print(solution_s),
                #print(s)
                s1 = copy.copy(s)
                s2 = copy.copy(s)

                s1[d] = s[d] - gamma
                s2[d] = s[d] + gamma
                
                with multiprocessing.Manager() as manager:
                
                    temp_solution_s1 = multiprocessing.Value('d')
                    temp_solution_s2 = multiprocessing.Value('d')

                    p1 = multiprocessing.Process(target=self.evaluationFitness_para, args=(s1,temp_solution_s1))
                    p2 = multiprocessing.Process(target=self.evaluationFitness_para, args=(s2,temp_solution_s2))

                    p1.start() 
                    p2.start()

                    p1.join() 
                    p2.join()
                    
                solution_s1 = temp_solution_s1.value
                solution_s2 = temp_solution_s2.value

                if(solution_s1 != float('Inf')):
                    self.NumberOfEvaluations += 1
                if(solution_s2 != float('Inf')):
                    self.NumberOfEvaluations += 1                     

                #solution_s1 = self.evaluationFitness(s1)
                #solution_s2 = self.evaluationFitness(s2)
                
               

                if solution_s1 <= solution_s2:
                    if solution_s1 < solution_s:
                        s = copy.copy(s1)
                        #print(s)
                        solution_s = solution_s1
                else:
                    if solution_s2 < solution_s:
                        s = copy.copy(s2)
                        #print(s)
                        solution_s = solution_s2
                d += 1
            if solution_s >= bestTempSolution_s:
                gamma = gamma * self.GAMMA_DECRESE_STEP
            else:
                #print(solution_s)
                #print(s)
                bestTempSolution_s = solution_s
                
        resultMap = {}
        resultMap["coordinates"] = s
        resultMap["solution"] = bestTempSolution_s

        return resultMap
        #print(solution_s)
        #print("IntensiveLocalSearch")
    
    def goAndMoveUp(self):
    
        result  = True
        nbSpheres = self.m_indexes[self.m_CurrentLevel - 1]



        while nbSpheres == (2*self.m_dimension):
            self.goUp()
            print(nbSpheres)
            #print("leo")
            #print(self.m_indexes)
            #print(self.m_CurrentLevel)

            nbSpheres = self.m_indexes.get(self.m_CurrentLevel - 1)
            #print(nbSpheres)

        #print("After while")
        if 	self.m_CurrentLevel == 1:
            result = False
        elif self.NumberOfEvaluations < self.m_stopCriterion:
            self.moveUp(nbSpheres)

        return result
    
    def goUp(self):
        self.m_CurrentLevel = self.m_CurrentLevel - 1
        return

    def goDown(self):
        self.m_CurrentLevel = self.m_CurrentLevel + 1
        return

    def moveUp(self,nbSphere):
        tempList_HyperLastLevel = []
        tempList_HyperLastLevel = self.m_stackMemory[self.m_CurrentLevel - 1]
        self.m_CurrentHyperSphere = tempList_HyperLastLevel[nbSphere]
        #nbSphere_visite += 1
        self.m_indexes[self.m_CurrentLevel-1] += 1 
    
    def evaluationFitness_para(self,solution,F):
        #print(solution)
        print("ENTER PARALLELIZED version of the evaluation Fitneess")
        solution_out_of_bounds = False
        
        #print(solution)
        for i in solution:
            if(i < self.m_lowerBound):
                F.value = float('Inf')
                solution_out_of_bounds = True
            if(i > self.m_upperBound):
                F.value = float('Inf')
                solution_out_of_bounds = True

        if(solution_out_of_bounds == False):
            #F.value = Function.Shifted_Rosenbrock(self.m_dimension, solution)
            #F = mymnist.mymnist(solution)
            #F = Function.Shifted_Ackley(self.m_dimension, solution)
            #F.value = np.mean(GM.trainWithKFOld(self.m_solution_architecture,solution,self.m_x_train,self.m_y_train)) 
            F.value = GM.trainWithoutKFOld(self.m_solution_architecture,solution,self.m_x_train,self.m_y_train, self.m_x_test, self.m_y_test)
            F.value = F.value * -1

            # f = open("./Results_MNIST_2D.txt", "a")
            f = open(constants.output_file, "a")
            f.write(str(F.value*-1)+""+str(solution)+"\n")
            #f.write(str(self.NumberOfEvaluations)+"\n")
            f.close()

        print("For : " + str(self.NumberOfEvaluations) + " : ", end='')
        print(F.value, end='')
        print(";", end='')
        print(solution)
        

    def evaluationFitness(self,solution):
        #print(solution)
        print("Enter Non-paralellized version of the evaluation Fitneess")
        F = None
        solution_out_of_bounds = False
        
        #print(solution)
        for i in solution:
            if(i < self.m_lowerBound):
                return float('Inf')
            if(i > self.m_upperBound):
                return float('Inf')

        if(solution_out_of_bounds == False):
            #F = Function.Shifted_Rosenbrock(self.m_dimension, solution)
            #F = mymnist.mymnist(solution)
            #F = Function.Shifted_Ackley(self.m_dimension, solution)
            #F = np.mean(GM.trainWithKFOld(self.m_solution_architecture,solution,self.m_x_train,self.m_y_train))
            F = GM.trainWithoutKFOld(self.m_solution_architecture,solution,self.m_x_train,self.m_y_train, self.m_x_test, self.m_y_test)        
            self.NumberOfEvaluations += 1
         
            # f = open("./Results_MNIST_2D.txt", "a")
            f = open(constants.output_file, "a")
            f.write(str(F)+""+str(solution)+"\n")
            #f.write(str(self.NumberOfEvaluations)+"\n")
            f.close()
        
        print("For : " + str(self.NumberOfEvaluations)+" : " , end='')
        print(F, end='')
        print(";", end='')
        print(solution)
        #return F
        return F*-1