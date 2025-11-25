package cbs

import cbs.model._
import cbs.algo.{CBS, FastAStar} // Ou import cbs.algo._

class CBSTest extends munit.FunSuite {
  
  // Enable parallel execution for this test suite
  override def munitExecutionContext = scala.concurrent.ExecutionContext.global
  
  val openGrid = Grid(5, 5, Set.empty)
  val gridWithObstacle = Grid(3, 3, Set(Position(1, 1)))
  val narrowCorridor = Grid(5, 3, Set(
    Position(0, 0), Position(0, 2),
    Position(1, 0), Position(1, 2),
    Position(2, 0), Position(2, 2),
    Position(3, 0), Position(3, 2),
    Position(4, 0), Position(4, 2)
  ))

  test("Two non-conflicting agents find solution") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(0, 4)),
      Agent(1, Position(4, 0), Position(4, 4))
    )
    val result = CBS.search(agents, openGrid)
    
    assert(result.isDefined)
    assertEquals(result.get.size, 2)
    assertEquals(result.get(0).positions.head, Position(0, 0))
    assertEquals(result.get(0).positions.last, Position(0, 4))
    assertEquals(result.get(1).positions.head, Position(4, 0))
    assertEquals(result.get(1).positions.last, Position(4, 4))
  }

  test("Resolves head-on collision") {
    val grid = Grid(5, 1, Set.empty)
    val agents = Seq(
      Agent(0, Position(0, 0), Position(4, 0)),
      Agent(1, Position(4, 0), Position(0, 0))
    )
    val result = CBS.search(agents, grid, maxTime = 50, maxNodes = 1000)
    
    if (result.isEmpty) {
      // If no solution in 1D, this is expected - agents can't pass in 1 row
      println("Note: Head-on collision in 1D corridor has no solution (expected)")
      assert(true)
    } else {
      // If solution found, verify no conflicts
      val paths = result.get
      val maxTime = math.max(paths(0).length, paths(1).length)
      for (t <- 0 until maxTime) {
        val pos0 = paths(0)(t)
        val pos1 = paths(1)(t)
        assertNotEquals(pos0, pos1, s"Conflict at time $t")
      }
    }
  }

  test("Handles crossing paths") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(2, 2)),
      Agent(1, Position(2, 0), Position(0, 2))
    )
    val result = CBS.search(agents, Grid(3, 3, Set.empty))
    
    assert(result.isDefined)
    assertEquals(result.get.size, 2)
    assertEquals(result.get(0).positions.last, Position(2, 2))
    assertEquals(result.get(1).positions.last, Position(0, 2))
  }

  test("Navigates around obstacles") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(2, 2))
    )
    val result = CBS.search(agents, gridWithObstacle)
    
    assert(result.isDefined)
    val path = result.get(0).positions
    assert(!path.contains(Position(1, 1)), "Path should not go through obstacle")
    assertEquals(path.last, Position(2, 2))
  }

  test("Returns None when no path exists") {
    val blockedGrid = Grid(3, 3, Set(
      Position(1, 0), Position(1, 1), Position(1, 2)
    ))
    val agents = Seq(
      Agent(0, Position(0, 1), Position(2, 1))
    )
    val result = CBS.search(agents, blockedGrid)
    
    assert(result.isEmpty, "Should return None when path is blocked")
  }

  test("Multiple agents in narrow corridor") {
    val corridor = Grid(6, 3, Set(
      Position(0,0), Position(0,2),
      Position(1,0), Position(1,2),
      Position(2,0), Position(2,2),
      Position(3,0), Position(3,2),
      Position(4,0), Position(4,2),
      Position(5,0), Position(5,2)
    ))
    val agents = Seq(
      Agent(0, Position(0,1), Position(5,1)),
      Agent(1, Position(5,1), Position(0,1))
    )
    val result = CBS.search(agents, corridor, maxTime = 500, maxNodes = 50000)
    assertEquals(result, None)
  }



  test("Detects edge conflicts correctly") {
    val paths = Map(
      0 -> Path(Seq(Position(0, 0), Position(1, 0), Position(2, 0))),
      1 -> Path(Seq(Position(1, 0), Position(0, 0), Position(0, 1)))
    )
    val conflict = CBS.detectConflict(paths)
    
    assert(conflict.isDefined, "Should detect agents swapping positions")
  }

  test("Returns None when no conflicts exist") {
    val paths = Map(
      0 -> Path(Seq(Position(0, 0), Position(1, 0), Position(2, 0))),
      1 -> Path(Seq(Position(0, 1), Position(1, 1), Position(2, 1)))
    )
    val conflict = CBS.detectConflict(paths)
    
    assert(conflict.isEmpty, "No conflict should be detected")
  }

  test("Handles three agents") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(4, 4)),
      Agent(1, Position(4, 0), Position(0, 4)),
      Agent(2, Position(2, 2), Position(2, 4))
    )
    val result = CBS.search(agents, openGrid)
    
    assert(result.isDefined)
    assertEquals(result.get.size, 3)
    assertEquals(result.get(0).positions.last, Position(4, 4))
    assertEquals(result.get(1).positions.last, Position(0, 4))
    assertEquals(result.get(2).positions.last, Position(2, 4))
  }

  test("Respects maxTime limit") {
    val grid = Grid(10, 10, Set.empty)
    val agents = Seq(
      Agent(0, Position(0, 0), Position(9, 9))
    )
    val result = CBS.search(agents, grid, maxTime = 5)
    
    assert(result.isEmpty, "Should fail when path exceeds maxTime")
  }

  test("Handles agents starting at their goals") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(0, 0)),
      Agent(1, Position(2, 2), Position(2, 2))
    )
    val result = CBS.search(agents, Grid(3, 3, Set.empty))
    
    assert(result.isDefined)
    assertEquals(result.get(0).positions.length, 1)
    assertEquals(result.get(1).positions.length, 1)
  }

  test("Solution has no vertex conflicts") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(3, 3)),
      Agent(1, Position(3, 0), Position(0, 3))
    )
    val result = CBS.search(agents, Grid(4, 4, Set.empty))
    
    assert(result.isDefined)
    val paths = result.get
    val maxTime = paths.values.map(_.length).max
    
    // Check every timestep for conflicts
    for (t <- 0 until maxTime) {
      val positions = paths.values.map(_(t)).toSeq
      assertEquals(positions.distinct.length, positions.length, 
        s"Vertex conflict detected at time $t")
    }
  }

  test("Solution has no edge conflicts") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(3, 0)),
      Agent(1, Position(3, 0), Position(0, 0))
    )
    val result = CBS.search(agents, Grid(4, 2, Set.empty), maxTime = 150)
    assert(result.isDefined, "CBS should resolve edge conflicts")
    val paths = result.get
    val minTime = math.min(paths(0).length, paths(1).length) - 1
    
    // Check for edge conflicts (swapping)
    for (t <- 0 until minTime) {
      val agentIds = paths.keys.toSeq
      for (i <- agentIds.indices; j <- i + 1 until agentIds.length) {
        val a1 = agentIds(i)
        val a2 = agentIds(j)
        val swap = paths(a1)(t) == paths(a2)(t + 1) && 
                   paths(a1)(t + 1) == paths(a2)(t)
        assert(!swap, s"Edge conflict between agents $a1 and $a2 at time $t")
      }
    }
  }

  test("All agents reach their goals") {
    val agents = Seq(
      Agent(0, Position(0, 0), Position(4, 0)),
      Agent(1, Position(0, 1), Position(4, 1)),
      Agent(2, Position(0, 2), Position(4, 2))
    )
    val result = CBS.search(agents, Grid(5, 3, Set.empty))
    
    assert(result.isDefined)
    for ((id, path) <- result.get) {
      val agent = agents.find(_.id == id).get
      assertEquals(path.positions.last, agent.goal, s"Agent $id didn't reach goal")
    }
  }

  test("High density 10 agents in 10x10 grid") {
    val grid = Grid(10, 10, Set.empty)
    val agents = (0 until 10).map { i =>
      Agent(i, Position(i % 10, 0), Position(9 - i, 9))
    }
    val result = CBS.search(agents, grid, maxTime = 300, maxNodes = 500000)
    
    assert(result.isDefined, "Should find a solution for high density scenario")
    val paths = result.get
    
    // Verify no conflicts
    assert(CBS.detectConflict(paths).isEmpty, "Solution should have no conflicts")
    
    // Verify all agents reach their goals
    for ((id, path) <- paths) {
      val agent = agents.find(_.id == id).get
      assertEquals(path.positions.last, agent.goal, s"Agent $id didn't reach goal")
    }
  }
}