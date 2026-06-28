import cv2
import random
import re
import string
import numpy as np
import os

from indigo import Indigo
from indigo.renderer import IndigoRenderer

from PIL import Image, ImageDraw, ImageFont
from chemistry import normalize_nodes
from constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


def add_functional_group(indigo, mol, debug=False):
    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol


def add_explicit_hydrogen(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(indigo, mol, smiles):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    if random.random() < 0.1:
        symb = f'({gen_rand_condensed()})'
    return symb


def get_rand_num():
    if random.random() < 0.9:
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        return '1' + str(random.randint(2, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        if i >= 1 and random.random() < 0.8:
            break
        tokens.append(get_rand_symb())
        tokens.append(get_rand_num())
    return ''.join(tokens)


def add_rand_condensed(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        part_b = re.search(r'\$.*\$', part_b).group(0)[1:-1]
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            # special cases with extension
            smiles = smiles.split(' ')[0]
        return mol, smiles


def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))


def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol


def get_graph(mol, image, shuffle_nodes=False, pseudo_coords=False):
    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        if pseudo_coords:
            x, y, z = atom.xyz()
        else:
            x, y = atom.coords()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    if pseudo_coords:
        coords = normalize_nodes(np.array(coords))
        h, w, _ = image.shape
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph


def generate_indigo_image(smiles, mol_augment=True, default_option=False, shuffle_nodes=False, pseudo_coords=False,
                          include_condensed=True, debug=False, image_width=250):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-label-mode', 'hetero')
    indigo.setOption('render-font-family', 'Arial')
    indigo.setOption('render-image-width', image_width)

    if not default_option:
        thickness = random.uniform(0.8, 2)  # limit the sum of the following two parameters to be smaller than 4
        indigo.setOption('render-relative-thickness', thickness)
        indigo.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        if random.random() < 0.5:
            indigo.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption('render-stereo-style', 'old')
        if random.random() < 0.2:
            indigo.setOption('render-atom-ids-visible', True)

    success = True
    try:
        mol = indigo.loadMolecule(smiles)
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            else:
                mol.aromatize()
            new_smiles = mol.canonicalSmiles()
            # add_comment(indigo)
            mol = add_explicit_hydrogen(indigo, mol)
            mol = add_rgroup(indigo, mol, new_smiles)
            if include_condensed:
                mol = add_rand_condensed(indigo, mol)
            mol = add_functional_group(indigo, mol, debug)
            mol = add_color(indigo, mol)
            mol, new_smiles = generate_output_smiles(indigo, mol)
        else:
            new_smiles = smiles

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph = get_graph(mol, img, shuffle_nodes, pseudo_coords)
    except Exception:
        img = None
        new_smiles = smiles
        graph = None
        success = False
        
    return img, new_smiles, graph, success



def split_text_into_lines(text):  
    words = text.split(' ')  
    num_words = len(words)  

    lines = []  
    start = 0  
    while start < num_words:   
        words_this_line = random.randint(3, 4) 
        line = ' '.join(words[start:start + words_this_line])  
        lines.append(line)
        start += words_this_line  
      
    return lines


def draw_cond_img(text, font_path_list):
    text = split_text_into_lines(text)

    font_list = []
    width_list, height_list, line_spacing_list = [], [], []
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for idx, line in enumerate(text):
        font_path = random.choice(font_path_list)
        font_size = random.randint(9, 12)  # (9, 14)
        font = ImageFont.truetype(font_path, font_size)
        font_list.append(font)
        bbox = draw.textbbox((0, 0), line, font=font)
        width_list.append(bbox[2]-bbox[0])
        height_list.append(bbox[3]-bbox[1])
        if idx != len(text):
            line_spacing = random.randint(1, 5)   # (1, 10)
            line_spacing_list.append(line_spacing)

    padding = 5
    image_width = max(width_list) + 2*padding
    image_height = sum(height_list) + sum(line_spacing_list) + 2*padding
    image = Image.new("RGB", (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    y = padding
    align = random.choice(['left', 'center', 'right'])
    text_color = (0, 0, 0) # random.choice([(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0)]) 
    for idx, line in enumerate(text):
        width, height = width_list[idx], height_list[idx]
        if align == 'center':
            x = (image.width - width) / 2
        elif align == 'right':
            x = image.width - width - padding
        else:
            x = padding
        
        # text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text((x, y), line, font=font_list[idx], fill=text_color)
        y += (height + line_spacing_list[idx])
    
    return image


def get_fonts(font_dir):
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith('.ttf') or file.endswith('.ttc'):
                fonts.append(os.path.join(root, file))

    return fonts