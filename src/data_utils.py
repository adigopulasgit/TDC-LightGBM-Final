from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES list to Morgan fingerprints (ECFP)."""
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros((n_bits,), dtype=int))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
    return np.array(features)
