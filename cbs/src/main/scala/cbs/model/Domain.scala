package cbs.model

case class Position(x: Int, y: Int) {
  def neighbors: List[Position] = List(
    Position(x + 1, y), Position(x - 1, y),
    Position(x, y + 1), Position(x, y - 1)
  )
  
  def manhattanDistance(other: Position): Int = 
    Math.abs(x - other.x) + Math.abs(y - other.y)
}

case class Agent(id: Int, start: Position, goal: Position)

// Contraintes corrigées (Vertex + Edge)
// Rappel rapide des types nécessaires dans Domain.scala
sealed trait Constraint { def agentId: Int; def time: Int }
case class VertexConstraint(agentId: Int, position: Position, time: Int) extends Constraint
case class EdgeConstraint(agentId: Int, from: Position, to: Position, time: Int) extends Constraint

case class Path(positions: Seq[Position]) {
  require(positions.nonEmpty, "Path cannot be empty")
  def length: Int = positions.length
  def apply(t: Int): Position = 
    if (t < positions.length) positions(t) else positions.last
}

case class Grid(width: Int, height: Int, obstacles: Set[Position]) {
  // Optimisation simple : utiliser un tableau booléen si la grille est statique
  // Pour l'instant, on garde votre Set mais on prépare la méthode isValid
  def isValid(pos: Position): Boolean =
    pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height && !obstacles.contains(pos)
}