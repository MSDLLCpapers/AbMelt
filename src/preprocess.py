#!/usr/bin/env python3

#    MIT License

#    COPYRIGHT (C) 2024 MERCK SHARP & DOHME CORP. A SUBSIDIARY OF MERCK & CO., 
#    INC., RAHWAY, NJ, USA. ALL RIGHTS RESERVED

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import os
import sys
import propka.run as pk
import re 
import warnings 
import tempfile 
import shutil 
import glob 
import logging 
import inspect

def parse_propka (pka):
    #Parse the output from propka and store the results of interest in lists
    result_pka_file = open(pka, "r")
    list_results = []
    for l in result_pka_file:
        if not l.strip():
            continue
        else:
            if len(l.strip().split()) == 22:
                list_results.append([l.strip().split()[0], l.strip().split()[1], l.strip().split()[2], l.strip().split()[3], l.strip().split()[6], l.strip().split()[8]])
    result_pka_file.close()
    return(list_results)

def convert_pkas(pkas, pH):
    # extract residue pkas (lys, arg, asp, glu, gln, his)
    # propka3 skipping first residue of chain A breaks ordering for other residue types
    # ^^^ FIX NEEDED (works on titratation of HIS for now) ^^^
    #LYS_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'LYS' and pkas[res][2] == 'A']
    #ARG_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'ARG' and pkas[res][2] == 'A']
    #ASP_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'ASP' and pkas[res][2] == 'A']
    #GLU_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'GLU' and pkas[res][2] == 'A']
    #GLN_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'GLN' and pkas[res][2] == 'A']
    HIS_A = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'HIS' and pkas[res][2] == 'A']
    #LYS_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'LYS' and pkas[res][2] == 'B']
    #ARG_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'ARG' and pkas[res][2] == 'B']
    #ASP_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'ASP' and pkas[res][2] == 'B']
    #GLU_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'GLU' and pkas[res][2] == 'B']
    #GLN_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'GLN' and pkas[res][2] == 'B']
    HIS_B = [pkas[res] for res in range(len(pkas)) if pkas[res][0] == 'HIS' and pkas[res][2] == 'B']
    #LYS_protonsA = [('1' if float(LYS_A[res][3]) >= pH else '0') for res in range(len(LYS_A))]
    #ARG_protonsA = [('1' if float(ARG_A[res][3]) >= pH else '0') for res in range(len(ARG_A))]
    #ASP_protonsA = [('1' if float(ASP_A[res][3]) >= pH else '0') for res in range(len(ASP_A))]
    #GLU_protonsA = [('1' if float(GLU_A[res][3]) >= pH else '0') for res in range(len(GLU_A))]
    #GLN_protonsA = [('1' if float(GLN_A[res][3]) >= pH else '0') for res in range(len(GLN_A))]
    HIS_protonsA = [('2' if float(HIS_A[res][3].strip("*")) >= pH else '0') for res in range(len(HIS_A))]
    #LYS_protonsB = [('1' if float(LYS_B[res][3]) >= pH else '0') for res in range(len(LYS_B))]
    #ARG_protonsB = [('1' if float(ARG_B[res][3]) >= pH else '0') for res in range(len(ARG_B))]
    #ASP_protonsB = [('1' if float(ASP_B[res][3]) >= pH else '0') for res in range(len(ASP_B))]
    #GLU_protonsB = [('1' if float(GLU_B[res][3]) >= pH else '0') for res in range(len(GLU_B))]
    #GLN_protonsB = [('1' if float(GLN_B[res][3]) >= pH else '0') for res in range(len(GLN_B))]
    HIS_protonsB = [('2' if float(HIS_B[res][3].strip("*")) >= pH else '0') for res in range(len(HIS_B))]
    return  HIS_protonsA + HIS_protonsB

def protonation_state (pdb, path, pH = 7.4):
    #Run Propka3 on pdb and return pKa summary
    pk.single(pdb, optargs=['--pH=%s'%(pH)],  stream=path, write_pka=True)
    pkas = parse_propka(os.path.splitext(pdb)[0]+'.pka')

    # remove NME because gromacs will add chain termini for CHARMM36m
    cli_cmd = 'grep -v " NME " '
    cli_cmd2 = ' > tmpfile && mv tmpfile '
    grep_cmd = cli_cmd + pdb + cli_cmd2 +pdb
    os.system(grep_cmd)

    # string of protonation states for residue types (lys, arg, asp, glu, his)
    gromacs_input = convert_pkas(pkas, pH)
    
    return gromacs_input

def canonical_index (pdb):
    from anarci import anarci
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    import re
    pdbparser = PDBParser()
    structure = pdbparser.get_structure(pdb, pdb)
    chains = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
    # annoatate light chain (currently A chain with MOE homology modeling)
    L_seq = [('L', chains['A'])]
    results_L = anarci(L_seq, scheme="imgt", output=False)
    numbering_L, alignment_details_L, hit_tables_L = results_L
    lc_anarci = [v for k, v in numbering_L[0][0][0]]
    lc_anarci_txt = ''.join(lc_anarci)
    lc_anarci_n = [k[0] for k, v in numbering_L[0][0][0]]
    gapl, cdr1l, cdr2l, cdr3l = [], [], [], []
    for i in range(0, len(lc_anarci)):
        if lc_anarci_n[i] >= 27 and lc_anarci_n[i] <= 38:
            cdr1l.append(i)
        elif lc_anarci_n[i] >= 56 and lc_anarci_n[i] <= 65:
            cdr2l.append(i)
        elif lc_anarci_n[i] >= 105 and lc_anarci_n[i] <= 117:
            cdr3l.append(i)
    for i in range(0, len(lc_anarci)):
        if lc_anarci[i] == '-':
            gapl.append(i)

    # convert imgt alignment indices back to pdb seq positions

    lc = chains['A']
    cdrll_imgt = [lc_anarci[res] for res in cdr1l]
    cdrll_imgt = ''.join(cdrll_imgt)
    cdrll_imgt = cdrll_imgt.replace('-','')
    #print(cdrll_imgt)
    cdr1l_pdb = [(match.start() + 1 , match.end()) for match in re.finditer(cdrll_imgt, lc)]

    cdr2l_imgt = [lc_anarci[res] for res in cdr2l]
    cdr2l_imgt = ''.join(cdr2l_imgt)
    cdr2l_imgt = cdr2l_imgt.replace('-','')
    #print(cdr2l_imgt)
    cdr2l_pdb = [(match.start() + 1, match.end()) for match in re.finditer(cdr2l_imgt, lc)]

    cdr3l_imgt = [lc_anarci[res] for res in cdr3l]
    cdr3l_imgt = ''.join(cdr3l_imgt)
    cdr3l_imgt = cdr3l_imgt.replace('-','')
    #print(cdr3l_imgt)
    cdr3l_pdb = [(match.start() + 1, match.end()) for match in re.finditer(cdr3l_imgt, lc)]

    lc_pdb = [(1, len(lc))]
    
    #print(cdr1l_pdb)
    #print(lc[cdr1l_pdb[0][0]])
    #print(lc[cdr1l_pdb[0][1]])

    #print(cdr2l_pdb)
    #print(lc[cdr2l_pdb[0][0]])
    #print(lc[cdr2l_pdb[0][1]])

    #print(cdr3l_pdb)
    #print(lc[cdr3l_pdb[0][0]])
    #print(lc[cdr3l_pdb[0][1]])


    #print(lc_anarci_txt)
    #print(cdr1l)
    #print(cdr2l)
    #print(cdr3l)
    #print(gapl)

    # annotate heavy chain (currently B chain with MOE homology modeling)
    H_seq = [('H', chains['B'])]
    results_H = anarci(H_seq, scheme="imgt", output=False)
    numbering_H, alignment_details_H, hit_tables_H = results_H
    hc_anarci = [v for k, v in numbering_H[0][0][0]]
    hc_anarci_txt = ''.join(hc_anarci)
    hc_anarci_n = [k[0] for k, v in numbering_H[0][0][0]]
    gaph, cdr1h, cdr2h, cdr3h = [], [], [], []
    for i in range(0, len(hc_anarci)):
        if hc_anarci_n[i] >= 27 and hc_anarci_n[i] <= 38:
            cdr1h.append(i)
        elif hc_anarci_n[i] >= 56 and hc_anarci_n[i] <= 65:
            cdr2h.append(i)
        elif hc_anarci_n[i] >= 105 and hc_anarci_n[i] <= 117:
            cdr3h.append(i)

    for i in range(0, len(hc_anarci)):
        if hc_anarci[i] == '-':
            gaph.append(i)
    
    # convert imgt alignment indices back to pdb seq positions

    hc = chains['B']
    cdrlh_imgt = [hc_anarci[res] for res in cdr1h]
    cdrlh_imgt = ''.join(cdrlh_imgt)
    cdrlh_imgt = cdrlh_imgt.replace('-','')
    #print(cdrlh_imgt)
    cdr1h_pdb = [(match.start() + 1 + len(lc), match.end() + len(lc)) for match in re.finditer(cdrlh_imgt, hc)]

    cdr2h_imgt = [hc_anarci[res] for res in cdr2h]
    cdr2h_imgt = ''.join(cdr2h_imgt)
    cdr2h_imgt = cdr2h_imgt.replace('-','')
    #print(cdr2h_imgt)
    cdr2h_pdb = [(match.start() + 1 + len(lc), match.end() + len(lc)) for match in re.finditer(cdr2h_imgt, hc)]

    cdr3h_imgt = [hc_anarci[res] for res in cdr3h]
    cdr3h_imgt = ''.join(cdr3h_imgt)
    cdr3h_imgt = cdr3h_imgt.replace('-','')
    #print(cdr3h_imgt)
    cdr3h_pdb = [(match.start() + 1 + len(lc), match.end() + len(lc)) for match in re.finditer(cdr3h_imgt, hc)]

    hc_pdb = [(1 + len(lc), len(hc) + len(lc))]
    
    #print(cdr1h_pdb)
    #print(hc[cdr1h_pdb[0][0]])
    #print(hc[cdr1h_pdb[0][1]])

    #print(cdr2h_pdb)
    #print(hc[cdr2h_pdb[0][0]])
    #print(hc[cdr2h_pdb[0][1]])

    #print(cdr3h_pdb)
    #print(hc[cdr3h_pdb[0][0]])
    #print(hc[cdr3h_pdb[0][1]])

    #print(hc_anarci_txt)
    #print(cdr1h)
    #print(cdr2h)
    #print(cdr3h)
    #print(gaph)


    annotation = [str('ri ' + str(lc_pdb[0][0]) + '-' + str(lc_pdb[0][1])), 'name 10 light_chain', 
                str('ri ' + str(hc_pdb[0][0]) + '-' + str(hc_pdb[0][1])), 'name 11 heavy_chain',
                str('ri ' + str(cdr1l_pdb[0][0]) + '-' + str(cdr1l_pdb[0][1])), 'name 12 cdr1l',
                str('ri ' + str(cdr2l_pdb[0][0]) + '-' + str(cdr2l_pdb[0][1])), 'name 13 cdr2l',
                str('ri ' + str(cdr3l_pdb[0][0]) + '-' + str(cdr3l_pdb[0][1])), 'name 14 cdr3l',
                str('ri ' + str(cdr1h_pdb[0][0]) + '-' + str(cdr1h_pdb[0][1])), 'name 15 cdr1h',
                str('ri ' + str(cdr2h_pdb[0][0]) + '-' + str(cdr2h_pdb[0][1])), 'name 16 cdr2h',
                str('ri ' + str(cdr3h_pdb[0][0]) + '-' + str(cdr3h_pdb[0][1])), 'name 17 cdr3h',
                str('12 | 13 | 14 | 15 | 16 | 17 '), 'name 18 cdrs',
                'q']

    return annotation

def edit_mdp(mdp, new_mdp=None, extend_parameters=None, **substitutions): 
    """Change values in a Gromacs mdp file. 
    
        Parameters and values are supplied as substitutions, eg ``nsteps=1000``. 
        
        By default the template mdp file is **overwritten in place**. 
    
        If a parameter does not exist in the template then it cannot be substituted 
        and the parameter/value pair is returned. The user has to check the 
        returned list in order to make sure that everything worked as expected. At 
        the moment it is not possible to automatically append the new values to the 
        mdp file because of ambiguities when having to replace dashes in parameter 
        names with underscores (see the notes below on dashes/underscores). 
        
        If a parameter is set to the value ``None`` then it will be ignored. 
        
        :Arguments: 
            *mdp* : filename 
                filename of input (and output filename of ``new_mdp=None``) 
            *new_mdp* : filename 
                filename of alternative output mdp file [None] 
            *extend_parameters* : string or list of strings 
                single parameter or list of parameters for which the new values  
                should be appended to the existing value in the mdp file. This  
                makes mostly sense for a single parameter, namely 'include', which 
                is set as the default. Set to ``[]`` to disable. ['include'] 
            *substitutions* 
                parameter=value pairs, where parameter is defined by the Gromacs mdp file;  
                dashes in parameter names have to be replaced by underscores. 
        
        :Returns:     
            Dict of parameters that have *not* been substituted. 
        
        **Example** :: 
        
            edit_mdp('md.mdp', new_mdp='long_md.mdp', nsteps=100000, nstxtcout=1000, lincs_iter=2) 
        
        .. Note:: 
        
            * Dashes in Gromacs mdp parameters have to be replaced by an underscore 
                when supplied as python keyword arguments (a limitation of python). For example 
                the MDP syntax is  ``lincs-iter = 4`` but the corresponding  keyword would be  
                ``lincs_iter = 4``. 
            * If the keyword is set as a dict key, eg ``mdp_params['lincs-iter']=4`` then one 
                does not have to substitute. 
            * Parameters *aa_bb* and *aa-bb* are considered the same (although this should 
                not be a problem in practice because there are no mdp parameters that only  
                differ by a underscore).  
            * This code is more compact in ``Perl`` as one can use ``s///`` operators: 
                ``s/^(\s*${key}\s*=\s*).*/$1${val}/`` 
        
        .. SeeAlso:: One can also load the mdp file with 
                    :class:`gromacs.formats.MDP`, edit the object (a dict), and save it again. 
        """ 
    base = '~/.gromacswrapper/templates/'
    if new_mdp is None: 
        new_mdp = mdp 
    if extend_parameters is None: 
        extend_parameters = ['include'] 
    else: 
        extend_parameters = list(asiterable(extend_parameters))
    
    mdp = base + mdp
    new_mdp = base + new_mdp

    def asiterable(v):
        if isinstance(v, str):
            return [v]
        try:
            iter(v)
            return v
        except TypeError:
            return [v]
        
    logger = logging.getLogger('gromacs.cbook') 
    # None parameters should be ignored (simple way to keep the template defaults) 
    substitutions = dict([(k,v) for k,v in substitutions.items() if not v is None]) 

    params = list(substitutions.keys()) # list will be reduced for each match
    # if values in substitions are list then join them with spaces
    substitutions = {k: '   '.join(str(x) for x in v) if isinstance(v, list) else str(v) for k, v in substitutions.items()}

    def demangled(p): 
        """Return a RE string that matches the parameter.""" 
        return p.replace('_', '[-_]')  # must catch either - or _ 

    #patterns = dict([(parameter, 
     #                   re.compile("""\ 
      #                  (?P<assignment>\s*%s\s*=\s*)  # parameter == everything before the value 
       #                 (?P<value>[^;]*)              # value (stop before comment=;) 
        #                (?P<comment>\s*;.*)?          # optional comment            
         #               """ % demangled(parameter), re.VERBOSE)) 
          #          for parameter in substitutions]) 
    patterns = dict([(parameter, 
                        re.compile("""(?P<assignment>\s*%s\s*=\s*)(?P<value>[^;]*)(?P<comment>\s*;.*)?""" % demangled(parameter), re.VERBOSE)) 
                    for parameter in substitutions]) 

    target = tempfile.TemporaryFile() 
    with open(os.path.expanduser(mdp)) as src: 
        logger.info("editing mdp = %r: %r" % (mdp, substitutions.keys())) 
        for line in src: 
            new_line = line.strip()  # \n must be stripped to ensure that new line is built without break 
            for p in params: 
                m = patterns[p].match(new_line)
                if m: 
                    # I am too stupid to replace a specific region in the string so I rebuild it 
                    # (matching a line and then replacing value requires TWO re calls) 
                    #print 'line:' + new_line 
                    #print m.groupdict() 
                    if m.group('comment') is None: 
                        comment = '' 
                    else: 
                        comment = " "+m.group('comment') 
                    assignment = m.group('assignment') 
                    if not assignment.endswith(' '): 
                        assignment += ' ' 
                    # build new line piece-wise: 
                    new_line = assignment 
                    if p in extend_parameters: 
                        # keep original value and add new stuff at end 
                        new_line += str(m.group('value')) + ' ' 
                    new_line += str(substitutions[p]) + comment 

                    params.remove(p) 
                    break
            new_line = new_line.encode('utf-8')
            target.write(new_line + b'\n') 
    target.seek(0) 
    # XXX: Is there a danger of corrupting the original mdp if something went wrong? 
    with open(os.path.expanduser(new_mdp), 'wb') as final: 
        shutil.copyfileobj(target, final) 
    target.close() 
    # return all parameters that have NOT been substituted 
    if len(params) > 0: 
        logger.warn("Not substituted in %(new_mdp)r: %(params)r" % vars()) 
    return dict([(p, substitutions[p]) for p in params]) 
