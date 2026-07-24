# Population-Based & Swarm Algorithms — Reading List (with summaries)

> **Purpose:** Survey of interesting problems solved with population-based / nature-inspired
> optimization (Genetic Algorithm, Ant Colony Optimization, Particle Swarm Optimization, Swarm
> Intelligence). Use this to (1) understand the foundations, (2) see "out-of-the-box" applications,
> and (3) find inspiration for a **new problem**.
>
> **Reading tip:** For every paper ask two questions — *"What is the chromosome / particle?"* and
> *"What is the fitness function?"* Those two answers give you ~80% of the paper.
>
> *Note: summaries below are condensed from each paper's abstract/idea so you don't have to open every
> link. For exact numbers/quotes, open the link before citing.*

---

## Table of Contents
1. [Foundational / Base Papers](#1-foundational--base-papers)
2. [Classic Benchmark-Application Base Papers](#2-classic-benchmark-application-base-papers)
3. [Out-of-the-Box Applications](#3-out-of-the-box-applications)
4. [Survey / Review Papers](#4-survey--review-papers)
5. [The Pattern Behind "Interesting"](#5-the-pattern-behind-interesting)

---

## 1. Foundational / Base Papers

**1. Holland — *Adaptation in Natural and Artificial Systems* (1975)** · GA origin
[MIT Press](https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/) · [Archive](https://archive.org/details/geneticalgorithm0000gold)
The book that founded the field of genetic algorithms. Holland frames adaptation as a search problem
and shows how biological operators — selection, **crossover**, **mutation**, inversion — can be made
into a mathematical algorithm. Introduces the idea of a *population of chromosomes* evolving over
generations and the famous **schema theorem** explaining why building blocks of good solutions spread.
Foundation for almost everything in evolutionary computation.

**2. Goldberg — *Genetic Algorithms in Search, Optimization & Machine Learning* (1989)** · GA textbook
[ACM](https://dl.acm.org/doi/10.5555/534133)
The classic tutorial that made GAs mainstream, written by Holland's student. Explains the full GA
mechanism step by step (encoding, roulette-wheel selection, crossover, mutation) with worked examples
and Pascal code. Covers benchmark problems like the **knapsack** and function optimization, plus theory
(schemata, building-block hypothesis). The reference most projects cite for "how a GA works."

**3. Kennedy & Eberhart — "Particle Swarm Optimization", IEEE ICNN (1995)** · PSO origin
[PDF](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)
The original PSO paper. A social psychologist and an engineer modelled **bird-flocking / fish-schooling**
behaviour into an optimizer. Each candidate solution is a *particle* with a position and velocity; it is
pulled toward its own best (**pbest**) and the swarm's best (**gbest**). Simple, few parameters, no
gradients needed — works well on continuous, non-linear functions. This is the source of the velocity
update formula on your slide.

**4. Shi & Eberhart — "A Modified Particle Swarm Optimizer" (1998)** · PSO inertia weight
[historical review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516836/)
Adds the **inertia weight `w`** to the velocity update — the term that controls how much a particle keeps
its previous direction. A large `w` favours **exploration** (roaming widely); a small `w` favours
**exploitation** (refining locally). Often decreased over time. This single tweak greatly improved PSO's
convergence and is in nearly every modern PSO variant (it's the `w` in your slide's equation).

**5. Clerc & Kennedy — "The Particle Swarm — Explosion, Stability and Convergence" (2002)** · PSO theory
[review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516836/)
The theoretical analysis of *why* PSO can blow up (velocities exploding) and how to stop it. Introduces
the **constriction factor**, a coefficient that mathematically guarantees the swarm converges instead of
diverging. Provides the stability conditions for choosing the learning factors `c1, c2`. The go-to
reference whenever PSO behaviour/stability is discussed.

**6. Dorigo, Maniezzo & Colorni — "Ant System", IEEE Trans. SMC-B (1996)** · ACO origin
[PDF](http://staff.washington.edu/paymana/swarm/dorigo96-itsmc.pdf)
The first Ant Colony Optimization algorithm. Artificial ants build solutions step by step, depositing
**pheromone** on good components; later ants probabilistically prefer high-pheromone choices, while
**evaporation** removes old trails to keep exploring. Demonstrated on the **Traveling Salesman Problem**.
Establishes the core ACO loop on your slide (construct → evaluate → deposit → evaporate → repeat).

**7. Dorigo & Gambardella — "Ant Colony System: ... to the TSP", IEEE TEC (1997)** · ACO improved
[PDF](http://faculty.washington.edu/paymana/swarm/dorigo97-itec.pdf)
An improved, much stronger ACO variant. Adds a **pseudo-random-proportional rule** (balances greedy vs
exploratory choices), **local pheromone updates** as ants move, and **global update only on the best
tour**. Significantly faster and better than the original Ant System on TSP, and the version most real
applications are based on.

**8. Dorigo & Di Caro — "The Ant Colony Optimization Metaheuristic" (1999)** · ACO generalized
[Springer](https://link.springer.com/chapter/10.1007/0-306-48056-5_9)
Generalizes ACO from "an algorithm for TSP" into a **general metaheuristic framework** that can attack
any combinatorial problem expressible as building a path on a graph. Defines the abstract components
(construction graph, pheromone model, heuristic info) so you can map *your own* problem onto ACO. Useful
template when adapting ACO to a new domain.

**9. Bonabeau, Dorigo & Theraulaz — *Swarm Intelligence: From Natural to Artificial Systems* (1999)** · SI book
The defining book for swarm intelligence as a field. Explains how **simple agents following simple local
rules** (ants, termites, bees, flocks) produce **emergent, intelligent group behaviour** with no central
controller. Bridges biology and engineering, covering ant foraging, division of labour, and self-
organization, then turns them into algorithms. Good for the conceptual "why swarms work" section.

**10. Storn & Price — "Differential Evolution" (1997)** · another population-based optimizer
[J. Global Optimization](https://link.springer.com/article/10.1023/A:1008202821328)
Introduces Differential Evolution (DE), a population-based optimizer for **continuous** problems. Creates
new candidates by adding the **scaled difference of two random population members** to a third, then
keeps the better of parent/child. Very simple, robust, few parameters — a strong alternative/companion to
GA and PSO worth knowing for comparison.

**11. "Comparative Analysis of Four ACO Variants (AS, Rank-AS, MMAS, ACS)" (2024)** · ACO survey
[arXiv](https://arxiv.org/pdf/2405.15397)
A modern, student-friendly comparison of the main ACO flavours: original **Ant System**, **Rank-based
AS**, **Max-Min AS** (clamps pheromone to a [min,max] range to avoid premature convergence), and **Ant
Colony System**. Explains how each handles the exploration/exploitation trade-off and benchmarks them.
Handy to understand which ACO variant to pick.

---

## 2. Classic Benchmark-Application Base Papers

**12. Traveling Salesman Problem (TSP)** · GA, ACO · (testbed in #6, #7)
The canonical benchmark for combinatorial optimization: find the shortest tour visiting every city once.
Representation = a **permutation of cities** (chromosome) or a path built by ants; fitness = `1/distance`.
Used to introduce both GA (permutation crossover/mutation) and ACO. If you invent a routing-style
problem, this is the base.

**13. Job-shop / production scheduling** · GA, ACO
[ACO metaheuristic chapter](https://link.springer.com/chapter/10.1007/0-306-48056-5_9)
Assign jobs/operations to machines over time to minimize total completion time (**makespan**) subject to
ordering and machine constraints. A classic NP-hard problem where GA and ACO both do well. Chromosome =
an ordering/assignment of operations; fitness = makespan or tardiness. Base for any "assign tasks to
resources" problem.

**14. Vehicle Routing Problem (VRP)** · ACO, GA
[ACO for real-world VRP (Springer)](https://link.springer.com/article/10.1007/s11721-007-0005-x)
Generalization of TSP: multiple vehicles serve many customers from a depot, minimizing total distance
under capacity/time-window constraints. Hugely practical (delivery, garbage collection, school buses).
Solution = set of routes; fitness penalizes distance + constraint violations. Base for delivery/logistics
projects.

**15. Feature selection for ML** · GA, PSO, ACO
[Binary ACO feature selection (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1568494616304057)
Pick the **smallest subset of features** that keeps (or improves) a model's accuracy — exactly the
`10110010` binary-string example on your slide. Chromosome = bitmask of features; fitness =
`accuracy − penalty × (#features)`. Reduces overfitting and training cost. Base for any "select the best
subset" problem.

**16. University / exam timetabling** · GA
[Exam scheduling with GA (arXiv)](https://arxiv.org/pdf/1902.01360)
Assign exams/classes to rooms and time-slots so no student/teacher/room clashes, respecting many soft
preferences. Heavily constrained, NP-hard, very relatable. Chromosome = an assignment of events to
slots; fitness = `−(hard violations × big + soft violations)`. Base for any scheduling project.

**17. Image segmentation / multilevel thresholding** · PSO
[Improved PSO multilevel thresholding (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0306283)
Find the set of intensity **thresholds** that best splits an image into regions. Brute force is
exponential in the number of thresholds, so PSO searches the threshold values, maximizing a criterion
like **Kapur entropy** or Otsu variance. Particle = a vector of thresholds. Base for image-processing
optimization.

**18. Network routing** · ACO
[ACO algorithms overview (Wikipedia + refs)](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)
Route data packets through a network along low-cost / low-congestion paths, adapting as the network
changes. ACO fits naturally because pheromone trails on links act like a distributed, self-updating
routing table (e.g. AntNet). Base for telecom / IoT routing problems.

**19. Knapsack / 0-1 combinatorial** · GA · (textbook example in #2)
Choose items (each with weight + value) to maximize value without exceeding a capacity. The simplest
"select a subset under a budget" problem and a standard first GA exercise. Chromosome = bitmask of items;
fitness = total value, with a penalty for exceeding capacity. Base for budget/resource-allocation
problems.

---

## 3. Out-of-the-Box Applications

### A. Hardware & Engineering Design *(output often looks "alien")*

**20. Evolved antenna for NASA ST5 satellite** · GA / Genetic Programming
[paper](http://alglobus.net/NASAwork/papers/Space2006Antenna.pdf) · [wiki](https://en.wikipedia.org/wiki/Evolved_antenna)
An evolutionary algorithm designed the X-band antenna for NASA's ST5 spacecraft (2006) — the **first
evolved hardware flown in space**. The chromosome encoded the antenna's bent-wire geometry; fitness =
how well it met gain/beamwidth/bandwidth requirements. The resulting odd "bent paperclip" shape matched
or beat the human-engineered design. The poster child for "AI designs things humans wouldn't."

**21. Wind-turbine layout via "design mining"** · Evolutionary Algorithm
[arXiv](https://arxiv.org/pdf/1410.0547)
Evolves the **placement of interacting wind turbines** in a farm so that wake/turbulence effects are
minimized and total energy capture is maximized. Because turbines disturb each other's airflow, the
search space is complex and non-linear. Solution = set of turbine coordinates; fitness = total farm
power output. Shows EAs designing physical layouts, not just tuning numbers.

**22. Antennas for ultra-high-energy neutrino detection** · Evolutionary Algorithm
[arXiv](https://arxiv.org/pdf/2005.07772)
Uses evolutionary search to design specialized antennas for particle-physics experiments that hunt for
ultra-high-energy neutrinos. Like the NASA case, the antenna geometry is evolved against a physics-based
performance simulation. Demonstrates EAs in cutting-edge scientific instrumentation where human intuition
about optimal shape is limited.

**23. Digital circuit design from high-level synthesis** · ACO (patented)
[USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8296712)
Applies ACO during chip design (high-level synthesis) to reduce **interconnection and multiplexing cost**
— i.e. how functional units are wired and shared. Ants build candidate hardware configurations; pheromone
reinforces low-cost wiring choices. Patented industrial use, showing ACO inside EDA (electronic design
automation) tools.

### B. Creativity & Art *(fitness is subjective / human-in-the-loop)*

**24. AMUSE: improvised jazz melody over a harmony** · GA
[paper](https://people.cs.nott.ac.uk/pszeo/docs/publications/amuse07.pdf)
A GA that **improvises a melody** over a given chord progression. Each chromosome encodes a sequence of
notes; the fitness function rewards musically desirable properties (fitting the harmony, smooth contour,
rhythm) so evolved melodies sound like plausible improvisation. Nice example of encoding a creative
artifact as a chromosome and designing a "musicality" fitness.

**25. GGA-MG: Generative Genetic Algorithm for Music Generation** · GA + LSTM
[arXiv](https://arxiv.org/pdf/2004.04687)
Combines a GA with an **LSTM neural network**: the LSTM acts as the fitness judge, scoring how
"human-like" a generated tune is, while the GA evolves the note sequences toward high scores. The hybrid
produces melodies that are both well-structured and similar to human compositions. Good template for
"GA + deep-learning fitness function."

**26. Interactive GA for creative design (fashion, faces, art)** · Interactive GA
[Springer](https://link.springer.com/article/10.1023/A:1013614519179)
In an **Interactive GA (IGA)**, the *human user is the fitness function* — they rate or pick the designs
they like, and the GA evolves toward their taste over rounds. Used for fashion design, evolving faces,
emotion-based image retrieval, and art. Key idea: optimize **subjective** goals that have no formula.
Perfect inspiration for a "personal preference" project.

**27. Evolutionary art mimicking human creativity** · Genetic Programming
[arXiv](https://arxiv.org/pdf/1001.1401)
Tries to bake **characteristics of human creativity** (novelty, complexity, aesthetic measures) directly
into the fitness function of an evolutionary art system, so it can generate appealing images *without*
constant human rating. Tackles the hard question: can you write a formula for "creative/beautiful"?
Good for thinking about automatic aesthetic fitness.

### C. Games & Strategy

**28. Evolving a grandmaster-level chess evaluation function** · GA + coevolution
[arXiv](https://arxiv.org/pdf/1711.08337)
A GA tunes the parameters of a chess engine's **evaluation function** (how it scores a board). Organisms
first evolve to **mimic grandmaster moves** from game databases, then improve further via
**coevolution** (engines play each other). The evolved program reportedly beat a two-time world
computer-chess champion. Shows GAs learning expert strategy, not just numbers.

**29. Genetically programmed chess-endgame strategies** · Genetic Programming
[ACM](https://dl.acm.org/doi/10.1145/1143997.1144144)
Uses Genetic Programming to **evolve whole strategies** (as programs/trees) for chess endgames. The
evolved players can draw or win against an expert hand-coded strategy and draw against the strong engine
CRAFTY. Demonstrates evolving *logic/behaviour*, not just parameters — the chromosome is an actual
program.

### D. Logistics, Routing & Disaster Relief *(hard real-world constraints)*

**30. ACO for real-world vehicle routing** · ACO
[Springer](https://link.springer.com/article/10.1007/s11721-007-0005-x)
Applies ACO to *actual* delivery operations: a Swiss supermarket chain (time windows), an Italian
distributor (pickup + delivery), and **online routing in Lugano** where orders arrive *during*
delivery. Shows ACO handling messy real constraints and even dynamic, changing problems — not just clean
textbook VRP.

**31. Disaster last-mile distribution with security convoys** · ACO
[ResearchGate](https://www.researchgate.net/publication/220058891_Ant_colony_optimization_for_real-world_vehicle_routing_problems)
Routes relief supplies in dangerous post-disaster zones (modelled on the **2010 Haiti earthquake** and
**2005 Niger famine**) where trucks must travel together in **convoys** for safety. That security
constraint is encoded into the fitness/cost. A striking example of bending a standard algorithm to a
life-or-death, constraint-heavy real scenario.

**32. Cultural-heritage tourist-route recommender** · improved ACO
[ACM 2025](https://dl.acm.org/doi/10.1145/3730436.3730499)
Builds personalized sightseeing routes through heritage sites, **balancing limited time and budget**
against how much a tourist wants to see. ACO (plus local search) constructs routes that maximize value
within constraints — a smart-city / sustainable-tourism twist on routing. Good model for a
"recommendation as optimization" idea.

### E. Scheduling & Timetabling

**33. Sports tournament timetabling** · metaheuristics incl. GA
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221724004338)
Schedules *who plays whom, when and where* in a league/tournament under fairness and venue constraints
(e.g. balanced home/away, no team plays twice in a row). Surveys which algorithm to pick. Chromosome = a
fixture assignment; fitness = constraint satisfaction + fairness. Relatable, demo-able scheduling
problem.

**34. Real-world train timetabling** · memetic / permutation EA
[arXiv](https://arxiv.org/pdf/cs/0510091)
A permutation-based evolutionary algorithm (with local search = "memetic") that builds **train
timetables** respecting track capacity, safety gaps, and priorities. Optimizes throughput / delay on a
real rail network. Shows EAs scaling to large, tightly-constrained transportation scheduling.

**35. Exam scheduling for central exams** · GA
[arXiv](https://arxiv.org/pdf/1902.01360)
Solves large central-exam scheduling with a GA: assign exams to slots/rooms so no student has clashing
exams and load is spread out. Chromosome = exam→slot mapping; fitness heavily penalizes clashes. A clean,
classic constrained-scheduling case study.

**36. Student timetabling respecting student preferences** · GA
[NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10280284/)
A GA that builds class timetables while honouring **student preferences** (e.g. free slots for sport,
training, jobs), not just clash-avoidance. Adds the human-preference dimension to scheduling. Good
inspiration if you want a "personalized schedule" project.

### F. Medicine & Biology

**37. Detecting hypoglycemia from EEG signals** · GA + neural net
[GAs in Medicine review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4678452/)
Combines a GA's global search with a neural network to **detect low blood sugar from EEG brain signals**.
The GA optimizes the network's weights/structure for better detection. Part of a broader review showing
GAs across radiology, oncology, cardiology, etc. Example of GA tuning a diagnostic model.

**38. Reinforced GA for structure-based drug design** · GA + reinforcement learning
[arXiv](https://arxiv.org/pdf/2211.16508)
Designs candidate **drug molecules** that fit a target protein's 3D binding pocket. A GA evolves
molecules, guided by reinforcement learning so the search is smarter than random mutation. Fitness =
predicted binding affinity + drug-likeness. Cutting-edge "AI for drug discovery" use of GA.

**39. Ligand-GA: automated protein-inhibitor design** · GA
[bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.11.463970.full.pdf)
A GA that automatically designs small-molecule **inhibitors** (ligands) for a given protein. Chromosome =
a molecular structure; fitness = docking score (how strongly/specifically it binds). Evolves better
binders over generations. Concrete, recent example of evolving chemistry.

**40. ACO for the HP protein-folding problem** · parallel ACO
[ResearchGate](https://www.researchgate.net/publication/301948364_Parallel_Ant_Colony_Optimization_for_the_HP_Protein_Folding_Problem)
Predicts how a protein folds in the simplified **HP lattice model** by having ants build folding
configurations; pheromone reinforces low-energy folds. Parallelized for speed. Folding is a huge
combinatorial search, making it a natural (and hard) ACO target. Bio + optimization crossover.

**41. Functional modules in protein–protein interaction networks** · improved ACO
[ResearchGate](https://www.researchgate.net/publication/262209445_Improved_Ant_Colony_Optimization_for_Detecting_Functional_Modules_in_Protein-Protein_Interaction_Networks)
Finds groups of proteins that work together (**functional modules / clusters**) inside a large
protein-interaction network. ACO searches the graph using a heuristic that mixes network topology with
biological function. A graph-clustering / community-detection use of ACO in bioinformatics.

**42. Brain-MRI tumor segmentation** · multidimensional PSO clustering
[MDPI](https://www.mdpi.com/1424-8220/25/9/2800)
Uses a multidimensional PSO that **automatically finds the right number of clusters** to segment brain
tumors in MRI scans, combining pixel intensity and spatial distance. Particle = a clustering
configuration; fitness = cluster quality. Beats fixed-k methods like plain K-means. Medical imaging via
swarm clustering.

**43. Cardiac image segmentation via active contours** · PSO
[NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3665182/)
Drives multiple **active contours ("snakes")** with PSO to outline the heart / left ventricle in CT and
MR images. PSO moves the contours to best fit organ boundaries, splitting the search space into polar
sections. Shows PSO guiding a classic computer-vision technique for medical use.

### G. Swarms, Networks & Data

**44. UAV swarm for weed detection in organic orchards (Agriculture 5.0)** · swarm intelligence
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2542660524003597)
A swarm of autonomous drones cooperatively scans an orange orchard to **detect and treat weeds**,
coordinating coverage without a central controller. Combines AI vision with swarm coordination for
precision, chemical-reducing agriculture. Real, literal "swarm intelligence" in the field.

**45. Swarm UAVs for search-and-rescue with thermal sensing** · bio-inspired swarm
[NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12835151/)
A bio-inspired drone swarm searches disaster areas using **thermal sensing** to find people, with
swarm-optimization coordinating who searches where for efficient coverage. Decentralized, robust to
individual drone failure. Strong example of swarms for humanitarian SAR.

**46. Maximizing wireless-sensor-network lifetime** · PSO
[ETASR](https://www.etasr.com/index.php/ETASR/article/view/6752)
Schedules which sensors are active vs sleeping (a **set-cover** problem) so the network keeps full
coverage while **batteries last as long as possible**. PSO searches activation schedules; fitness =
network lifetime under coverage constraints. Practical IoT energy-optimization case.

**47. Steering-stability control of 4-wheel-drive EVs** · PSO-tuned neural controller
[Frontiers](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2024.1378175/full)
Uses PSO to tune a **neural-network + PID controller** that keeps a four-wheel-drive electric vehicle
stable while steering on different terrains. PSO optimizes the controller gains/weights offline. Example
of PSO in real-time control-systems engineering.

**48. Mining gradual patterns from data** · ACO
[arXiv](https://arxiv.org/pdf/2208.14795)
Applies ACO to **data mining** — discovering "gradual patterns" of the form *"the more X, the more Y"*
in datasets. Ants search the space of attribute combinations, pheromone reinforcing strong correlations.
Shows ACO as a knowledge-discovery tool, not just routing.

**49. Learning Bayesian network structure** · ACO
[arXiv](https://arxiv.org/pdf/1401.3464)
Uses ACO to search for the best **structure of a Bayesian network** (which variables cause/depend on
which) — a notoriously huge search space. Ants build candidate graphs; fitness = how well the structure
explains the data. A machine-learning model-discovery use of ACO.

**50. Energy-hub management** · PSO
[Scientific Reports](https://www.nature.com/articles/s41598-024-76010-y)
A hybrid PSO optimizes how a residential **energy hub** dispatches electricity, heating, and cooling
across non-linear equipment constraints to cut cost/waste. Particle = an operating schedule of the energy
devices; fitness = total cost / efficiency. Smart-grid / energy-systems application.

---

## 4. Survey / Review Papers
*(great for the intro / related-work section of your report)*

**51. "A review on genetic algorithm: past, present, and future" (2021)**
[Springer](https://link.springer.com/article/10.1007/s11042-020-10139-6)
Broad survey of GAs: history, operators, variants, and application areas across engineering, ML, and
science. Good single source to cite for "GA background and landscape."

**52. "Cumulative Major Advances in PSO from 2018 to present" (2024)**
[Springer](https://link.springer.com/article/10.1007/s11831-024-10185-5)
Up-to-date survey of PSO variants, theoretical analysis, and modern applications. Use it to position your
project within recent PSO work.

**53. PSO: a historical review up to current developments**
[NIH/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516836/)
Traces PSO from the 1995 origin through inertia weight, constriction factor, and beyond. Good for the
"evolution of the algorithm" narrative.

**54. "Swarm Intelligence-Based Multi-Robotics: A Comprehensive Review" (2024)**
[MDPI](https://www.mdpi.com/2673-9909/4/4/64)
Reviews how swarm intelligence is used to coordinate multi-robot / drone systems (search-and-rescue,
agriculture, exploration). Good for swarm-robotics framing.

**55. "Swarm Intelligence in Healthcare" (research-trend mapping)**
[NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11416823/)
Maps research trends applying swarm intelligence to healthcare — diagnosis, treatment optimization,
medical imaging, patient monitoring. Useful if you lean toward a medical application.

---

## 5. The Pattern Behind "Interesting"

A problem feels *out of the box* when it has one or more of these twists:

| Twist | Example above |
|-------|---------------|
| Output looks **non-human / surprising** | NASA antenna (#20) |
| Fitness is **subjective / human-in-the-loop** | Interactive GA art & fashion (#26) |
| A **weird real-world constraint** is encoded | Convoy routing in disaster zones (#31) |
| Applied to a **domain nobody associates with optimization** | Music (#24), chess (#28), weed-spraying (#44) |
| Optimizes **something hard to measure** | "Generalization", "musicality", "aesthetics" (#27) |

**Next step:** Pick a direction (everyday/quirky · creative/subjective · tech-ML · biology), then design
a *new* problem with one of the twists above + a clear chromosome/particle and a clear fitness function.

---
*Compiled for Lab 3 — Population-Based Approach. 55 works across foundations, benchmarks, out-of-the-box
applications, and surveys, each with a short summary.*
