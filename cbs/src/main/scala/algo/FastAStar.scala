package cbs.algo

import cbs.model._
import scala.collection.mutable

object FastAStar {
  
  private case class SearchState(
    position: Position,
    time: Int,
    costFromStart: Int,      // g-score
    estimatedTotalCost: Int, // f-score
    path: Vector[Position]
  )
  
  def findPath(
    agent: Agent,
    grid: Grid,
    constraints: Set[Constraint],
    heuristicWeight: Double = 1.0,
    maxTime: Int = 200 // Augmenté par défaut pour les grands grids
  ): Option[Path] = {
    
    // 1. Optimisation : Pré-calcul des contraintes pour accès O(1)
    val vertexConstraints = mutable.HashSet[(Int, Int, Int)]()
    val edgeConstraints = mutable.HashSet[(Int, Int, Int, Int, Int)]() // x1,y1 -> x2,y2 @ t

    // On calcule le temps de la dernière contrainte pour savoir combien de temps attendre au but
    var minEndTime = 0

    constraints.filter(_.agentId == agent.id).foreach {
      case VertexConstraint(_, pos, t) => 
        vertexConstraints.add((pos.x, pos.y, t))
        minEndTime = math.max(minEndTime, t)
      case EdgeConstraint(_, from, to, t) => 
        edgeConstraints.add((from.x, from.y, to.x, to.y, t))
        minEndTime = math.max(minEndTime, t)
    }
    
    // 2. Fonction de vérification rapide
    def isConstrained(currPos: Position, nextPos: Position, nextTime: Int): Boolean = {
      // Vertex: Interdit d'être sur la case au temps t
      if (vertexConstraints.contains((nextPos.x, nextPos.y, nextTime))) return true
      // Edge: Interdit de faire le mouvement A->B au temps t
      if (edgeConstraints.contains((currPos.x, currPos.y, nextPos.x, nextPos.y, nextTime))) return true
      false
    }
    
    def heuristic(pos: Position): Int = 
      (pos.manhattanDistance(agent.goal) * heuristicWeight).toInt
    
    // 3. Optimisation "Tie-Breaking" : Préférer les nœuds avec le plus grand g-score (costFromStart)
    // Cela force A* à aller tout droit vers le but au lieu d'explorer en largeur inutilement.
    val openSet = mutable.PriorityQueue.empty[SearchState](
      using Ordering.fromLessThan[SearchState] { (a, b) =>
        if (a.estimatedTotalCost == b.estimatedTotalCost) {
          a.costFromStart < b.costFromStart // Plus grand g-score est prioritaire (Max-Heap inversé)
        } else {
          a.estimatedTotalCost > b.estimatedTotalCost // Plus petit f-score est prioritaire
        }
      }
    )

    val visitedStates = mutable.Map[(Int, Int, Int), Int]()
    
    // Initialisation
    val startH = heuristic(agent.start)
    val initialState = SearchState(agent.start, 0, 0, startH, Vector(agent.start))
    openSet.enqueue(initialState)
    visitedStates((agent.start.x, agent.start.y, 0)) = 0
    
    // Directions (Haut, Bas, Gauche, Droite, Attendre)
    // Tableaux statiques pour éviter l'allocation d'objets List à chaque itération
    val dx = Array(0, 0, 1, -1, 0)
    val dy = Array(1, -1, 0, 0, 0)

    while (openSet.nonEmpty) {
      val current = openSet.dequeue()
      
      // 4. Correction "Lazy Goal" : On ne s'arrête que si on a dépassé toutes les contraintes futures
      if (current.position == agent.goal && current.time >= minEndTime) {
        return Some(Path(current.path))
      }
      
      if (current.time < maxTime) {
        val nextTime = current.time + 1
        val nextCost = current.costFromStart + 1
        
        // Boucle optimisée sur les voisins (incluant l'attente)
        var i = 0
        while (i < 5) {
          val nx = current.position.x + dx(i)
          val ny = current.position.y + dy(i)
          
          // Vérification manuelle des limites pour éviter de créer des objets Position invalides
          if (nx >= 0 && nx < grid.width && ny >= 0 && ny < grid.height) {
            
            // On ne crée l'objet Position que si nécessaire
            val nextPos = Position(nx, ny)
            
            // Vérification Obstacle + Contraintes CBS
            if (!grid.obstacles.contains(nextPos) && !isConstrained(current.position, nextPos, nextTime)) {
              
              val h = heuristic(nextPos)
              val f = nextCost + h
              
              val stateKey = (nx, ny, nextTime)
              val previousCost = visitedStates.getOrElse(stateKey, Int.MaxValue)
              
              if (nextCost < previousCost) {
                visitedStates(stateKey) = nextCost
                
                openSet.enqueue(SearchState(
                  position = nextPos,
                  time = nextTime,
                  costFromStart = nextCost,
                  estimatedTotalCost = f,
                  path = current.path :+ nextPos
                ))
              }
            }
          }
          i += 1
        }
      }
    }
    
    None
  }
}