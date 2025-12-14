
"""Tools for interacting with MolSysMT."""

from __future__ import annotations


def get_info(pdb_id: str) -> str:
    """Load a molecular system from a PDB ID and get basic information.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB ID of the system to load.

    Returns
    -------
    str
        A formatted string with basic information about the system, or an
        error message if the system could not be loaded.
    """
    try:
        import molsysmt as msm  # type: ignore

        molsys = msm.load(f"pdbid:{pdb_id}")
        info = msm.get(molsys, n_atoms=True, n_groups=True, n_chains=True, n_molecules=True)
        return (
            f"Successfully loaded PDB ID {pdb_id}:\n"
            f"- {info['n_atoms']} atoms\n"
            f"- {info['n_groups']} groups\n"
            f"- {info['n_chains']} chains\n"
            f"- {info['n_molecules']} molecules"
        )
    except ModuleNotFoundError as exc:
        if exc.name == "molsysmt":
            return (
                "MolSysMT is not available in this Python environment.\n"
                "Install MolSysSuite tools locally (often via conda) and run the agent from that environment.\n"
                "Tip: `molsys-ai tools doctor` prints a recommended setup."
            )
        return f"Missing dependency: {exc}"
    except Exception as e:
        return f"Failed to load PDB ID {pdb_id}. Error: {e}"
