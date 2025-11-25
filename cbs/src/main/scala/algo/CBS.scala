package cbs.algo

import cbs.model._
import scala.collection.mutable
import scala.util.boundary
import scala.util.boundary.break

// Node interne pour l'arbre de recherche CBS
case class CBSNode(paths: Map[Int, Path], constraints: Set[Constraint], cost: Int)

object CBS {
  
  // Définition interne des conflits
  sealed trait Conflict
  case class VertexConflict(a1: Int, a2: Int, pos: Position, time: Int) extends Conflict
  case class EdgeConflict(a1: Int, a2: Int, prev1: Position, curr1: Position, prev2: Position, curr2: Position, time: Int) extends Conflict

  def search(
      agents: Seq[Agent],
      grid: Grid,
      weight: Double = 1.0,
      maxTime: Int = 100,
      maxNodes: Int = 10000
  ): Option[Map[Int, Path]] = {
    
    // Cache pour éviter de recalculer A* si les contraintes d'un agent n'ont pas changé
    val pathCache = mutable.Map[(Int, Set[Constraint]), Option[Path]]()
    
    def getPath(agent: Agent, constraints: Set[Constraint]): Option[Path] = {
      // On ne garde que les contraintes pertinentes pour cet agent pour la clé de cache
      val agentConstraints = constraints.filter(_.agentId == agent.id)
      pathCache.getOrElseUpdate(
        (agent.id, agentConstraints),
        FastAStar.findPath(agent, grid, agentConstraints, weight, maxTime)
      )
    }
    
    // 1. Chemins initiaux sans contraintes
    val rootPathsMap = mutable.Map[Int, Path]()
    for (agent <- agents) {
      getPath(agent, Set.empty) match {
        case Some(p) => rootPathsMap(agent.id) = p
        case None => return None // Impossible dès le départ
      }
    }
    val rootPaths = rootPathsMap.toMap
    
    val root = CBSNode(rootPaths, Set.empty, rootPaths.values.map(_.length).sum)
    val open = mutable.PriorityQueue.empty[CBSNode](using Ordering.by(-_.cost))
    open.enqueue(root)
    
    var expanded = 0
    
    while (open.nonEmpty && expanded < maxNodes) {
      val node = open.dequeue()
      expanded += 1
      
      detectConflict(node.paths) match {
        case None => 
          return Some(node.paths) // Solution valide trouvée !
        
        case Some(conflict) =>
          // Génération des contraintes selon le type de conflit
          val newConstraints: Seq[Constraint] = conflict match {
            case VertexConflict(a1, a2, pos, time) =>
              Seq(
                VertexConstraint(a1, pos, time),
                VertexConstraint(a2, pos, time)
              )
            case EdgeConflict(a1, a2, p1_prev, p1_curr, p2_prev, p2_curr, time) =>
              // Pour un swap, on interdit la TRAVERSÉE spécifique
              Seq(
                EdgeConstraint(a1, p1_prev, p1_curr, time),
                EdgeConstraint(a2, p2_prev, p2_curr, time)
              )
          }

          // Création des nœuds enfants
          for (constraint <- newConstraints) {
            val agentId = constraint.agentId
            val updatedConstraints = node.constraints + constraint
            val agent = agents.find(_.id == agentId).get
            
            // Re-planifier SEULEMENT l'agent concerné
            getPath(agent, updatedConstraints) match {
              case Some(newPath) =>
                val newPaths = node.paths.updated(agentId, newPath)
                val newCost = newPaths.values.map(_.length).sum
                
                // Ajouter l'enfant si le chemin est valide
                open.enqueue(CBSNode(newPaths, updatedConstraints, newCost))
              case None =>
                // Branche morte (pas de chemin pour cet agent avec cette contrainte)
            }
          }
      }
    }
    None
  }

  def detectConflict(paths: Map[Int, Path]): Option[Conflict] = {
    boundary {
      val agentIds = paths.keys.toSeq.sorted
      val maxPathLength = paths.values.map(_.length).max
      
      // On vérifie chaque instant t
      for (time <- 0 until maxPathLength) {
        for (i <- agentIds.indices; j <- i + 1 until agentIds.length) {
          val a1 = agentIds(i)
          val a2 = agentIds(j)
          
          // Position actuelle
          val pos1 = paths(a1)(time)
          val pos2 = paths(a2)(time)
          
          // 1. Vertex Conflict
          if (pos1 == pos2) {
            break(Some(VertexConflict(a1, a2, pos1, time)))
          }
          
          // 2. Edge Conflict (Swap) : Nécessite de regarder t et t-1
          if (time > 0) {
            val prev1 = paths(a1)(time - 1)
            val prev2 = paths(a2)(time - 1)
            
            // Si A1 est sur l'ancienne case de A2 ET A2 est sur l'ancienne case de A1
            if (pos1 == prev2 && pos2 == prev1) {
              break(Some(EdgeConflict(a1, a2, prev1, pos1, prev2, pos2, time)))
            }
          }
        }
      }
      None
    }
  }
}