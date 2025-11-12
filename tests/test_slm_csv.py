import configparser
import logging
import sys
import os
import pytest
from rdflib import URIRef

from SPARQLLM.config import ConfigSingleton
from SPARQLLM.udf.SPARQLLM import reset_store, store
from SPARQLLM.utils.utils import print_result_as_table
from rdflib.plugins.sparql.operators import register_custom_function

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="module")
def setup_config():
    """
    Setup function to initialize the ConfigSingleton with an in-memory config.
    This ensures a clean configuration for testing.
    """
    ConfigSingleton.reset_instance()
    reset_store()

    # Création d'un objet ConfigParser en mémoire
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity for option names
    config['Associations'] = {
        'SLM-CSV': 'SPARQLLM.udf.mycsv.slm_csv',
        'SLM-FILE': 'SPARQLLM.udf.absPath.absPath'
    }

    # Instanciation de la configuration
    config_instance = ConfigSingleton(config_obj=config)
    config_instance.print_all_values()

    return config_instance  # Retourne l'instance pour une éventuelle utilisation dans les tests


@pytest.mark.skipif(not os.path.exists("./data/results.csv"), reason="CSV file './data/results.csv' not found.")
def test_sparql_csv_function(setup_config):
    """
    Test that the SPARQL function correctly processes CSV data.
    """
    query_str = """
    PREFIX ex: <http://example.org/>
    SELECT ?x ?z WHERE {
        BIND(ex:SLM-FILE("./data/results.csv") AS ?value)
        BIND(ex:SLM-CSV(?value) AS ?g)
        graph ?g {
            ?x <http://example.org/city> ?z .
        }
    } limit 10
    """
    result = store.query(query_str)

    # Ensure result is not empty
    assert result is not None, "SPARQL query returned None"

    # Convert result to a list for assertion checks
    rows = list(result)

    # Ensure that some rows are returned
    assert len(rows) > 0, "SPARQL query returned no results"

    # Debug output (optional)
    # print_result_as_table(result)


@pytest.mark.skipif(not os.path.exists("./data/results.csv"), reason="CSV file './data/results.csv' not found.")
def test_sparql_csv_with_mappings(setup_config):
    """
    Test that the SPARQL function correctly processes CSV data
    using an external CONSTRUCT mapping file (advanced use case).
    """

    # Create mapping file
    mappings_dir = "./data/mappings"
    mappings_file = os.path.join(mappings_dir, "mycsv.map")
    os.makedirs(mappings_dir, exist_ok=True)

    # Define the mapping query
    mapping_query = """
    PREFIX ex: <http://example.org/>
    PREFIX mycsv: <http://mycsv.org/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    CONSTRUCT {
        ?record mycsv:city-of-soccer ?cityValue .
        ?record mycsv:team-name ?teamValue .
    }
    WHERE {
        ?record rdf:type ex:Record .
        ?record ex:city ?cityValue .

        OPTIONAL { ?record ex:team ?teamValue . }
    }
    """

    try:
        with open(mappings_file, 'w') as f:
            f.write(mapping_query)
    except IOError as e:
        pytest.fail(f"Failed to create mapping file: {e}")

    query_str = """
    PREFIX ggf: <http://example.org/>
    PREFIX ex: <http://example.org/>
    PREFIX mycsv: <http://mycsv.org/>

    # Using SLM-CSV with external mapping file
    SELECT ?x ?z WHERE {
        BIND(ex:SLM-FILE("./data/results.csv") as ?value)
        BIND(ex:SLM-FILE("./data/mappings/mycsv.map") as ?mappings)
        BIND(ex:SLM-CSV(?value, ?mappings) AS ?g)

        graph ?g {
            ?x mycsv:city-of-soccer ?z .
        }
    } limit 10
    """

    result = store.query(query_str)

    assert result is not None, "SPARQL query with mapping returned None"

    rows = list(result)

    assert len(rows) > 0, "SPARQL query with mapping returned no results"

    # Debug output (optional)
    # print("\n--- Results for Mapped CSV Query ---")
    # print_result_as_table(result)

    # try:
    #     os.remove(mappings_file)
    # except OSError as e:
    #     logging.warning(f"Could not clean up mapping file: {e}")


if __name__ == "__main__":
    # to see test with logs...
    # pytest --log-cli-level=DEBUG tests/test_slm_csv.py
    pytest.main([sys.argv[0]])