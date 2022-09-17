import copy as copy
import Hypersphere as H
import Kiss as Kiss
import time as time

def getKeyFitness(a):
	return a.m_fitness

def getKeyRation(a):
	return a.m_bestRatio

"""hyper = H.Hypersphere()
hyper.m_radius = 3	

hyper2 = copy.deepcopy(hyper)

print(hyper.m_radius)
print(hyper2.m_radius)

hyper.m_radius = 4

print(hyper.m_radius)
print(hyper2.m_radius)"""


def run(solution_architecture,dim,train_data,train_labels, test_data, test_labels):
    fini = True
    nbSphere_visite = 0
    m_intNbFoisEnterLevelMax = 0
    print("DIMENSION IS "+str(dim))

    start_time = time.time()

    kiss = Kiss.Kiss(dim, 2, 1, 0,solution_architecture,train_data,train_labels,test_data, test_labels)

    while(kiss.NumberOfEvaluations < kiss.m_stopCriterion and fini == True):		

        print("START DECOMPOSITIONHYPERSPHERE")
        listHyper = kiss.DecompositionHyperSphere(kiss.m_CurrentHyperSphere)

        boucleListSubSphere = 0
        print("START EVALUATING HYPERSPHERE")
        while(boucleListSubSphere < (kiss.m_dimension * 2)):
            listHyper[boucleListSubSphere].inflateHyperSphere()
            listHyper[boucleListSubSphere].computeFitness(kiss)
            ##print(listHyper[boucleListSubSphere])
            boucleListSubSphere += 1


        #Gestion des lites de spheres
        #kiss.m_stackMemory[0] = 22

        if kiss.m_stackMemory.get(kiss.m_CurrentLevel) != None:
            del kiss.m_stackMemory[kiss.m_CurrentLevel]

        #Verification empty stack
        """
        if kiss.m_stackMemory.get(0) == None:
            print "Dict is Empty"
        else:
            print "Dickt is not Empty"
        """

        #Trier par fitness afin de mettre a jour la meilleur solution de navigation
        listHyper.sort(key=getKeyFitness)

        newFitness = listHyper[0].m_fitness
        currentFitness = kiss.m_CurrentHyperSphere.m_fitness

        if newFitness < currentFitness:
            kiss.m_bestSolutionCoordinatesNavigation = copy.copy(listHyper[0].m_fitnessCoordinates)
            kiss.m_bestSolutionFitnessNavigation = newFitness


        #Now sort by Rati (the bigger the better)	
        listHyper.sort(key=getKeyRation, reverse = True)

        kiss.m_stackMemory[kiss.m_CurrentLevel] = copy.deepcopy(listHyper)
        #See about managing the indexes of the stack of hyperspheres
        kiss.m_indexes[kiss.m_CurrentLevel] = 0

        #Remplacer la currentHypersphere
        """
        if kiss.m_stackMemory.get(kiss.m_CurrentLevel) == None:
            print "Dict is Empty"
        else:
            print "Dickt is not Empty"
        """
        #print(type())
        H = kiss.m_stackMemory.get(kiss.m_CurrentLevel)[0]
        kiss.m_CurrentHyperSphere = copy.copy(H)
        nbSphere_visite += 1

        print("*******"),
        print(kiss.m_CurrentLevel)

        kiss.m_indexes[kiss.m_CurrentLevel] += 1
        if kiss.m_CurrentLevel == kiss.m_kLevelsMax:
            #print("level max reached")

            print("START ENTER LEVEL 5")

            boucleSearchLevel5 = 0

            tempList_HyperLastLevel = kiss.m_stackMemory.get(kiss.m_CurrentLevel)
            nbSphere_visite = nbSphere_visite - 1
            while boucleSearchLevel5 < (2*kiss.m_dimension) and (kiss.NumberOfEvaluations < kiss.m_stopCriterion):
                m_intNbFoisEnterLevelMax += 1
                #print("Boucle Search Level 5")
                kiss.m_CurrentHyperSphere = tempList_HyperLastLevel[boucleSearchLevel5]

                nbSphere_visite += 1 

                tempSolution = []

                resultSolution = {}
                resultSolution = kiss.IntensiveLocalSearch()

                #print(resultSolution["coordinates"])

                if resultSolution["solution"] < kiss.m_bestSolutionFitness:
                    kiss.m_bestSolutionCoordinates = copy.copy(resultSolution["coordinates"])
                    kiss.m_bestSolutionFitness = resultSolution["solution"]

                #kiss.NumberOfEvaluations = 0;

                boucleSearchLevel5 += 1

                kiss.m_indexes[kiss.m_CurrentLevel] += 1
                #print(kiss.m_indexes[kiss.m_CurrentLevel])

            kiss.m_indexes[kiss.m_CurrentLevel] = 0
            fini = None
            fini = kiss.goAndMoveUp()

            if fini == False:
                print("ARBRE FINIT : Nb Sphere visite : ")
                print(nbSphere_visite)
        else:
            kiss.goDown()


    print("leo")
    print(kiss.nb_usefulDic)
    print(kiss.m_bestSolutionFitness)
   
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return([kiss.m_bestSolutionFitness,kiss.m_bestSolutionCoordinates])

