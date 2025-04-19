# the SLM-RECURSE function

The `SLM-RECURSE` UDF  allows recursive SPARQL queries based on CONSTRUCT queries. It implements point-fix semantic ie. recursion occurs as long as new elements are added. `SLM-RECURSE`ensures termination.


The `queries/simple-recurse.sparql` illustrates a simple transitive closure. Please notice that unlike property paths queries, it is possible to perform transitive closure on unbounded predicate.

The `queries/directory_recurse` combines recursion and UDF on the file system. The query recurse on the file system and construct a knowledge graph representing a subtree of the file system with size of files. It allows to easily compute the biggest file included in the subtree.


# SLM-RECURSE-UPDATE functions

The `SLM-RECURSE-UPDATE`function allows recursive SPARQL queries based on UPDATE queries. This allow to build turing complete program but does not ensure termination.

The `queries/tape_update.sparql`  executes a turing program declared in the `data/tape.nq` NQUAD file. The `tape.nq`represent in RDF a turing tape in named graph (with head state) and the state automata of the program in another named graph. The program just replace 1 by 0 and 0 by 1 on a turing tape.  The `tape_update`query allow to execute the program. The query stop when no insert/delete are performed after 1 recursion ie. when the turing program made no transition.