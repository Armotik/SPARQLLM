from rdflib import Literal
from rdflib.namespace import XSD
import logging

logger = logging.getLogger(__name__)

def PRINT(value):
    """
    UDF: PRINT(?value)
    - Effet de bord: écrit la valeur sur la sortie (logger + stdout)
    - Retourne la valeur (écho) comme Literal pour pouvoir être liée en SPARQL.

    Usage SPARQL (exemples):
      BIND(ex:PRINT(?txt) AS ?txt2)
      # ou si on ne veut pas réutiliser la valeur:
      BIND(ex:PRINT(?txt) AS ?_)
    """
    try:
        s = "" if value is None else str(value)
    except Exception:
        s = repr(value)

    # Log + stdout immédiat
    logger.info(f"PRINT: {s}")
    try:
        print(s, flush=True)
    except Exception:
        # Éviter toute propagation si stdout n'est pas dispo
        pass

    # Renvoie le même Literal si déjà un Literal, sinon un Literal(xsd:string)
    if isinstance(value, Literal):
        return value
    return Literal(s, datatype=XSD.string)
