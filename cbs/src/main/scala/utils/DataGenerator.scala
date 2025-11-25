package cbs.utils

import cbs.model._
import cbs.algo.CBS
import java.io.{File, PrintWriter, FileWriter}
import scala.util.Random
import scala.concurrent.{Future, Await, ExecutionContext}
import scala.concurrent.duration._
import java.util.concurrent.atomic.AtomicInteger

object DataGenerator {

  // Utilise tous les cœurs disponibles
  implicit val ec: ExecutionContext = ExecutionContext.global

  /**
   * Génère UN scénario et retourne sa chaîne JSON (ou None si échec)
   * Cette fonction est thread-safe (pas d'effets de bord).
   */
  def generateSingleSample(
      id: Int,
      minGrid: Int, maxGrid: Int,
      minAgents: Int, maxAgents: Int,
      obsDensity: Double
  ): Option[String] = {
    val rand = new Random() // Random est thread-safe en Scala pour des instances locales
    val w = rand.between(minGrid, maxGrid + 1)
    val h = rand.between(minGrid, maxGrid + 1)
    val n = rand.between(minAgents, maxAgents + 1)

    try {
      val (grid, agents) = generateScenario(w, h, n, obsDensity)
      // Timeout court pour ne pas bloquer les threads sur des cas impossibles
      CBS.search(agents, grid, maxTime = 100, maxNodes = 5000) match {
        case Some(paths) => Some(formatAsJson(id, w, h, grid.obstacles, agents, paths))
        case None => None
      }
    } catch {
      case _: Exception => None
    }
  }

  /**
   * Génération Parallèle par Batchs
   */
  def generateDataset(
      outputFile: String,
      targetSamples: Int,
      batchSize: Int = 1000 // Taille du bloc à traiter en RAM
  ): Unit = {
    
    println(s"=== Démarrage de la génération parallèle ===")
    println(s"Cible: $targetSamples échantillons")
    println(s"Threads: ${Runtime.getRuntime.availableProcessors()}")

    // Initialiser le fichier (écraser l'ancien)
    val writer = new PrintWriter(new File(outputFile))
    writer.write("[\n")
    writer.flush()

    var totalGenerated = 0
    val globalIdCounter = new AtomicInteger(0)
    val startTime = System.nanoTime()

    // Boucle jusqu'à avoir assez d'échantillons
    while (totalGenerated < targetSamples) {
      
      // 1. Créer une liste de tâches (Futures)
      val futures: Seq[Future[Option[String]]] = (0 until batchSize).map { _ =>
        Future {
          generateSingleSample(
            id = globalIdCounter.getAndIncrement(),
            minGrid = 8, maxGrid = 16, 
            minAgents = 2, maxAgents =10, 
            obsDensity = 0.2
          )
        }
      }

      // 2. Convertir Seq[Future[T]] en Future[Seq[T]]
      val batchFuture = Future.sequence(futures)

      // 3. Attendre la fin du batch (Bloquant mais nécessaire pour l'écriture fichier)
      // On donne 1 minute par batch max
      val results = Await.result(batchFuture, 2.minutes)

      // 4. Filtrer les succès et écrire
      val validJsons = results.flatten
      
      validJsons.zipWithIndex.foreach { case (json, idx) =>
        writer.write(json)
        // Ajouter une virgule sauf si c'est vraiment le tout dernier du dataset global
        if (totalGenerated + idx < targetSamples - 1) {
          writer.write(",\n")
        } else {
          writer.write("\n")
        }
      }
      writer.flush() // Forcer l'écriture disque

      totalGenerated += validJsons.size
      
      val progress = (totalGenerated.toDouble / targetSamples) * 100
      val elapsed = (System.nanoTime() - startTime) / 1e9
      val rate = totalGenerated / elapsed
      
      print(f"\rProgress: $totalGenerated / $targetSamples ($progress%.1f%%) - Vitesse: $rate%.1f items/sec")
      
      // Si on a dépassé la cible avec ce batch, on arrête
      if (totalGenerated >= targetSamples) {
        println("\nCible atteinte !")
      }
    }

    writer.write("]") // Fin du JSON array
    writer.close()
    println(s"\nDataset sauvegardé dans $outputFile")
  }

  // --- Méthodes utilitaires (copiées de la version précédente) ---
  
  def generateScenario(w: Int, h: Int, numAgents: Int, obstacleDensity: Double): (Grid, Seq[Agent]) = {
    val rand = new Random()
    val totalCells = w * h
    val numObstacles = (totalCells * obstacleDensity).toInt
    val obstacles = scala.collection.mutable.Set[Position]()
    
    while (obstacles.size < numObstacles) {
      obstacles.add(Position(rand.nextInt(w), rand.nextInt(h)))
    }
    
    val grid = Grid(w, h, obstacles.toSet)
    val validPositions = (for {
      x <- 0 until w; y <- 0 until h
      p = Position(x, y) if !obstacles.contains(p)
    } yield p).toVector

    if (validPositions.length < numAgents * 2) throw new RuntimeException("Too dense")
    val shuffled = rand.shuffle(validPositions)
    val agents = (0 until numAgents).map(i => Agent(i, shuffled(i), shuffled(i + numAgents)))
    (grid, agents)
  }

  private def formatAsJson(id: Int, w: Int, h: Int, obs: Set[Position], agents: Seq[Agent], paths: Map[Int, Path]): String = {
    def pStr(p: Position) = s"[${p.x},${p.y}]"
    val obsStr = obs.map(pStr).mkString("[", ",", "]")
    val agentsStr = agents.map(a => s"""{"id":${a.id},"start":${pStr(a.start)},"goal":${pStr(a.goal)}}""").mkString("[", ",", "]")
    val pathsStr = paths.map { case (aid, path) => s""""$aid": ${path.positions.map(pStr).mkString("[", ",", "]")}""" }.mkString("{", ",", "}")
    
    s"""  { "id": $id, "width": $w, "height": $h, "obstacles": $obsStr, "agents": $agentsStr, "paths": $pathsStr }"""
  }
}