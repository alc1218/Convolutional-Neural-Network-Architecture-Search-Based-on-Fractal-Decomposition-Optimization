import copy as copy
import math as math
import multiprocessing 

class Hypersphere:

	INFLATED_RATE = 1.75
	EPSILON_DIVISION_0 = 1 *10** -20;
	
	m_dimension = None
	m_center = []
	m_radius = None		
	
	m_fitnessCoordinates = []
	m_fitness = None

	m_bestRatio = -1;

	def __init__(self):
		print("HyperSphere created")
		
	def __init__(self, dimension,tempCenter, tempRadius):
		self.m_dimension = dimension
		self. m_center = copy.copy(tempCenter)
		self.m_radius = tempRadius		
		#print("HyperSphere created dimension,tempCenter, tempRadius")
		
		
	def inflateHyperSphere(self):
		#print(self.m_radius)
		self.m_radius = self.m_radius * self.INFLATED_RATE
		#print(self.m_radius)
		
	def computeFitness(self,context):
        
		#s1 & s2
		s1 = copy.copy(self.m_center)
		s2 = copy.copy(self.m_center)

		for i in range(0,context.m_dimension):
			s1[i] = self.m_center[i] + (self.m_radius / math.sqrt(context.m_dimension))
			s2[i] = self.m_center[i] - (self.m_radius / math.sqrt(context.m_dimension))

		with multiprocessing.Manager() as manager:
			temp_solution_s = multiprocessing.Value('d')
			temp_solution_s1 = multiprocessing.Value('d')
			temp_solution_s2 = multiprocessing.Value('d')
            
			p1 = multiprocessing.Process(target=context.evaluationFitness_para, args=(self.m_center,temp_solution_s))
    
			p2 = multiprocessing.Process(target=context.evaluationFitness_para, args=(s1,temp_solution_s1))
    
			p3 = multiprocessing.Process(target=context.evaluationFitness_para, args=(s2,temp_solution_s2))
        
			p1.start() 
			p2.start()
			p3.start()
    
			p1.join() 
			p2.join()
			p3.join()
            
		solution_s = temp_solution_s.value
		solution_s1 = temp_solution_s1.value
		solution_s2 = temp_solution_s2.value
        
		if(solution_s != float('Inf')):
			context.NumberOfEvaluations += 1
		if(solution_s1 != float('Inf')):
			context.NumberOfEvaluations += 1
		if(solution_s2 != float('Inf')):
			context.NumberOfEvaluations += 1
        
        
		#solution_s = context.evaluationFitness(self.m_center)
		#solution_s1 = context.evaluationFitness(s1)
		#solution_s2 = context.evaluationFitness(s2)
        
		distance_s = self.calculateDistance(self.m_center, context.m_bestSolutionCoordinatesNavigation)
		
		ratio_s = -1
		ration_s = abs(solution_s / (distance_s + self.EPSILON_DIVISION_0))
		
		#Traiter si la solution est == exp(100)
		#c-a-d en dehors des limite du domaine
		
		distance_s1 = self.calculateDistance(s1, context.m_bestSolutionCoordinatesNavigation)
		ratio_s1 = abs(solution_s1 / (distance_s1 + self.EPSILON_DIVISION_0))
		
		
		#s2

		
		distance_s2= self.calculateDistance(s2, context.m_bestSolutionCoordinatesNavigation)
		ratio_s2 = abs(solution_s2 / (distance_s2 + self.EPSILON_DIVISION_0))
		
		#fitness
		self.m_fitness = min(solution_s,solution_s1,solution_s2)

		#print(solution_s)
		#print(solution_s1)
		#print(solution_s2)
		#print("***")
		
		#ration
		self.m_bestRatio = max(ratio_s,ratio_s1,ratio_s2)
		
		#Get best ration
		if self.m_fitness == solution_s:
			self.m_fitnessCoordinates = copy.copy(self.m_center)
		elif self.m_fitness == solution_s1:
			self.m_fitnessCoordinates = copy.copy(s1)
		else:
			self.m_fitnessCoordinates = copy.copy(s2)
			
		if self.m_fitness < context.m_bestSolutionFitness:
			context.m_bestSolutionCoordinates = copy.copy(self.m_fitnessCoordinates)
			context.m_bestSolutionFitness = copy.copy(self.m_fitness)


	def calculateDistance(self,p1,p2):
		tempSum = 10
		result = None
		
		for i in range(0,self.m_dimension):
			tempSum = tempSum + (p1[i] - p2[i]) **2
		
		result = math.sqrt(tempSum)
		
		return result
	
	def __str__(self): 		
		"""Methode permettant d'afficher plus joliment notre objet""" 		
		for i in range(0,self.m_dimension): 			
			print("Dim["), 			
			print(i), 			
			print("] : "), 			
			print(self.m_center[i])
		print(self.m_fitness)	
		print(self.m_bestRatio)
		return ""	