package cbs

import munit.FunSuite
import cbs.model._
import cbs.algo.FastAStar

class FastAStarTest extends FunSuite {
  
  override def munitExecutionContext = scala.concurrent.ExecutionContext.global
  
  // Test grids
  val openGrid = Grid(5, 5, Set.empty)
  val gridWithWall = Grid(5, 5, Set(
    Position(2, 0), Position(2, 1), Position(2, 2), Position(2, 3)
  ))
  
  // ============================================================================
  // Basic Pathfinding Tests
  // ============================================================================
  
  test("finds straight path without obstacles") {
    val agent = Agent(0, Position(0, 0), Position(4, 0))
    val result = FastAStar.findPath(agent, openGrid, Set.empty)
    
    assert(result.isDefined, "Should find a path")
    val path = result.get
    assertEquals(path.positions.head, Position(0, 0), "Path should start at agent start")
    assertEquals(path.positions.last, Position(4, 0), "Path should end at agent goal")
    assertEquals(path.length, 5, "Path length should be 5 for distance 4")
  }
  
  test("finds diagonal path") {
    val agent = Agent(0, Position(0, 0), Position(3, 3))
    val result = FastAStar.findPath(agent, openGrid, Set.empty)
    
    assert(result.isDefined)
    val path = result.get
    assertEquals(path.positions.last, Position(3, 3))
    // Manhattan distance is 6, so path length is 7 (including start)
    assertEquals(path.length, 7)
  }
  
  test("navigates around wall obstacle") {
    val agent = Agent(0, Position(0, 2), Position(4, 2))
    val result = FastAStar.findPath(agent, gridWithWall, Set.empty)
    
    assert(result.isDefined, "Should find path around wall")
    val path = result.get
    assertEquals(path.positions.last, Position(4, 2))
    
    // Path should not go through wall
    val wallPositions = Set(Position(2, 0), Position(2, 1), Position(2, 2), Position(2, 3))
    assert(
      path.positions.forall(pos => !wallPositions.contains(pos)),
      "Path should not go through wall"
    )
  }
  
  test("returns None when no path exists") {
    val blockedGrid = Grid(5, 5, Set(
      Position(2, 0), Position(2, 1), Position(2, 2), 
      Position(2, 3), Position(2, 4)
    ))
    val agent = Agent(0, Position(0, 2), Position(4, 2))
    val result = FastAStar.findPath(agent, blockedGrid, Set.empty)
    
    assert(result.isEmpty, "Should return None when goal is unreachable")
  }
  
  // ============================================================================
  // Constraint Tests (CORRIGÉS)
  // ============================================================================
  
  test("avoids position blocked by constraint at specific time") {
    val agent = Agent(0, Position(0, 0), Position(2, 0))
    // OLD: Constraint(...) -> NEW: VertexConstraint(...)
    val constraints: Set[Constraint] = Set(VertexConstraint(0, Position(1, 0), 1))
    val result = FastAStar.findPath(agent, openGrid, constraints)
    
    assert(result.isDefined)
    val path = result.get
    assertNotEquals(
      path.positions(1), 
      Position(1, 0),
      "Should not be at constrained position at time 1"
    )
    assertEquals(path.positions.last, Position(2, 0), "Should still reach goal")
  }
  
  test("can wait in place to avoid constraint") {
    val narrowGrid = Grid(3, 1, Set.empty)
    val agent = Agent(0, Position(0, 0), Position(2, 0))
    // Force wait
    val constraints: Set[Constraint] = Set(VertexConstraint(0, Position(1, 0), 1))
    val result = FastAStar.findPath(agent, narrowGrid, constraints)
    
    assert(result.isDefined, "Should find path by waiting")
    val path = result.get
    
    assertNotEquals(path.positions(1), Position(1, 0))
    assert(path.length >= 4, "Path should be at least length 4 due to waiting")
  }

  test("respects multiple constraints simultaneously") {
    val agent = Agent(0, Position(0, 0), Position(2, 0))
    val constraints: Set[Constraint] = Set(
      VertexConstraint(0, Position(1, 0), 1),
      VertexConstraint(0, Position(0, 1), 1),
      VertexConstraint(0, Position(1, 1), 2)
    )
    val result = FastAStar.findPath(agent, openGrid, constraints)
    
    assert(result.isDefined)
    val path = result.get
    
    assertNotEquals(path.positions(1), Position(1, 0))
    assertNotEquals(path.positions(1), Position(0, 1))
  }
  
  // ============================================================================
  // Edge Cases (Lazy Goal & Time Limits)
  // ============================================================================
  
  test("respects maxTime limit") {
    val agent = Agent(0, Position(0, 0), Position(4, 4))
    // Max time 3 is too short for distance 8
    val result = FastAStar.findPath(agent, openGrid, Set.empty, maxTime = 3)
    assert(result.isEmpty, "Should return None when path exceeds maxTime")
  }
  
  test("WAITS at goal if future constraint exists (Lazy Goal fix)") {
    val agent = Agent(0, Position(0, 0), Position(1, 0)) // Distance 1
    // Constraint on Goal at time 5. Agent arrives at time 1.
    // Agent must wait at goal: t1(ok), t2(ok), t3(ok), t4(ok), t5(BLOCKED), t6(ok)
    // Wait... if goal is blocked at t5, it means agent cannot BE at goal at t5.
    val constraints: Set[Constraint] = Set(VertexConstraint(0, Position(1, 0), 2))
    
    val result = FastAStar.findPath(agent, openGrid, constraints)
    
    assert(result.isDefined)
    val path = result.get
    // Path: (0,0) -> (0,0)[Wait] -> (1,0)
    // Or: (0,0) -> (1,0) ... wait constraint violation ...
    
    // With Lazy Goal logic:
    // If constraint is at t=2 on Goal.
    // Path length must ensure we are safe AFTER t=2.
    // Actually, if we arrive at t=1, we are at Goal. At t=2 we are still at Goal.
    // So we verify that the path correctly handles this.
    
    val posAt2 = path(2) // Position at time 2
    assertNotEquals(posAt2, Position(1, 0), "Should not be at goal at time 2 due to constraint")
    assertEquals(path.positions.last, Position(1, 0))
  }
}