"""Native XeLaTeX manuscript output with preserved inline and display mathematics."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pymupdf


MM = 72 / 25.4
ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
           '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
           '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def escape_text(text):
    """Escape prose while preserving explicitly delimited inline mathematics."""
    text = str(text)
    if len(re.findall(r'(?<!\\)\$', text)) % 2:
        raise ValueError('Unbalanced inline mathematics delimiter')
    parts = re.split(r'(\$[^$]*\$)', text)
    rendered = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            rendered.append(part)
        else:
            rendered.append(''.join(ESCAPES.get(char, char) for char in part))
    return ''.join(rendered)


class LatexManuscript:
    def __init__(self, results, root, references):
        self.results, self.root, self.reference_registry = Path(results), Path(root), references
        self.tex, self.markdown, self.placements = [], [], []
        self.citation_order, self.equations = {}, []

    def format(self, value):
        parts = re.split(r'(\$[^$]*\$)', str(value))
        normalized = []
        for part in parts:
            # Legacy negative exponents need braces in existing math and math
            # delimiters in prose; never introduce nested dollar delimiters.
            if part.startswith('$') and part.endswith('$'):
                part = re.sub(r'10\^(-?\d+)\b', lambda m: '10^{'+m[1]+'}', part)
            else:
                part = re.sub(r'(?<!\w)10\^(-?\d+)\b', lambda m: '$10^{'+m[1]+'}$', part)
            normalized.append(part)
        text = ''.join(normalized)
        def remap(match):
            numbers = []
            for token in match[1].replace(' ', '').replace('–', '-').split(','):
                if '-' in token:
                    lo, hi = map(int, token.split('-'))
                    numbers.extend(range(lo, hi+1))
                else:
                    numbers.append(int(token))
            if 0 in numbers:
                return match[0]
            for old in numbers:
                if old not in self.reference_registry:
                    raise ValueError(f'Unknown reference {old}')
                self.citation_order.setdefault(old, len(self.citation_order)+1)
            return 'CITATIONTOKEN'+','.join(str(self.citation_order[n]) for n in numbers)+'ENDTOKEN'
        marked = re.sub(r'\[([0-9,\-– ]+)\]', remap, text)
        markdown = re.sub(r'CITATIONTOKEN([0-9,]+)ENDTOKEN', r'[\1]', marked)
        latex = re.sub(r'CITATIONTOKEN([0-9,]+)ENDTOKEN', r'\\textsuperscript{\1}', escape_text(marked))
        return markdown, latex

    def para(self, text, style='Body'):
        markdown, latex = self.format(text)
        if style == 'Title':
            latex = r'{\raggedright\fontsize{21}{24}\selectfont\bfseries '+latex+r'\par}\vspace{3mm}'
        elif style == 'Meta':
            latex = r'{\sffamily\footnotesize\color{gray} '+latex+r'\par}\vspace{3mm}'
        elif style == 'Summary':
            latex = r'{\bfseries '+latex+r'\par}\vspace{3mm}'
        elif style == 'Subheading':
            latex = r'\subsection*{'+latex+'}'
        elif style == 'Section':
            latex = r'\section*{'+latex+'}'
        else:
            latex += '\n\n'
        self.tex.append(latex)
        prefix = {'Title':'# ', 'Subheading':'### ', 'Section':'## '}.get(style, '')
        self.markdown.append(prefix+markdown+'\n')

    def section(self, text, newpage=False):
        if newpage:
            self.new_page()
        self.para(text, 'Section')

    def new_page(self):
        self.tex.append(r'\clearpage')

    def equation(self, latex, label, description=None):
        if label in {row['label'] for row in self.equations}:
            raise ValueError(f'Duplicate equation label: {label}')
        self.tex.append('\n\\begin{equation}\n\\label{'+label+'}\n'+latex+'\n\\end{equation}\n')
        self.markdown.append('$$\n'+latex+'\n$$\n')
        self.equations.append(dict(label=label, latex=latex))
        if description:
            self.para(description)

    def figure(self, name, label, caption, height=170*MM, width=183*MM):
        png = self.results/'figures'/name
        vector = png.with_suffix('.pdf')
        source = vector if vector.exists() else png
        if not source.exists():
            raise FileNotFoundError(source)
        markdown, latex = self.format(caption)
        relative = source.relative_to(self.root).as_posix()
        self.tex.append(r'\begin{figure}[H]'+'\n'+r'\centering'+'\n'+
                        r'\includegraphics[width='+f'{width/MM:.3f}'+r'mm,height='+f'{height/MM:.3f}'+
                        r'mm,keepaspectratio]{\detokenize{'+relative+'}}\n'+
                        r'\caption*{\textbf{'+escape_text(label)+r' | }'+latex+'}\n'+r'\end{figure}')
        self.markdown.append(f'![{label}: {markdown.split(". ")[0]}](../{png.relative_to(self.root)})\n')
        self.markdown.append(f'**{label} |** {markdown}\n')
        self.placements.append(dict(label=label, path=relative, width_mm=width/MM, max_height_mm=height/MM))

    def table(self, label, caption, headers, rows):
        markdown, latex = self.format(caption)
        cells = lambda values: ' & '.join(escape_text(v) for v in values)+r' \\'
        self.tex.append(r'\begin{table}[H]\centering\sffamily\footnotesize'+ '\n'+
                        r'\caption*{\textbf{'+escape_text(label)+r' | }'+latex+'}'+ '\n'+
                        r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.18}'+ '\n'+
                        r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n'+r'\toprule'+ '\n'+
                        cells(headers)+'\n'+r'\midrule'+ '\n'+
                        '\n'.join(cells(row) for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}')
        self.markdown.append(f'**{label} |** {markdown}\n')
        self.markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
                             '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')

    def references(self):
        self.section('References', newpage=True)
        self.tex.append(r'\begin{enumerate}[label=\arabic*.,leftmargin=6mm,itemsep=2mm]\small')
        for old, new in sorted(self.citation_order.items(), key=lambda item:item[1]):
            entry = self.reference_registry[old]
            chunks = re.split(r'(https?://\S+|[a-f0-9]{40})', entry)
            formatted = []
            for chunk in chunks:
                if chunk.startswith(('https://', 'http://')):
                    url = chunk.rstrip('.,;')
                    formatted.append(r'\url{'+url+'}'+escape_text(chunk[len(url):]))
                elif re.fullmatch(r'[a-f0-9]{40}', chunk):
                    formatted.append(r'\texttt{'+r'\allowbreak{}'.join(chunk[i:i+8] for i in range(0,len(chunk),8))+'}')
                else:
                    formatted.append(escape_text(chunk))
            self.tex.append(r'\item '+''.join(formatted))
            self.markdown.append(f'{new}. {entry}\n')
        self.tex.append(r'\end{enumerate}')

    def build(self, output, title):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        tex_path = self.root/'report/report.tex'
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        preamble = r'''\documentclass[11pt,a4paper]{article}
\usepackage[left=13.5mm,right=13.5mm,top=16mm,bottom=18mm]{geometry}
\usepackage{amsmath,booktabs,graphicx,longtable,array,caption,float,placeins,xcolor,enumitem}
\usepackage{fontspec,unicode-math}
\setmainfont{TeX Gyre Termes}
\setsansfont{Liberation Sans}
\setmathfont{Latin Modern Math}
\usepackage{microtype}
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{3.5mm}{2mm}
\titlespacing*{\subsection}{0pt}{3mm}{1mm}
\captionsetup{font={small,sf},skip=3mm,justification=raggedright,singlelinecheck=false}
\setlength{\parindent}{0pt}
\setlength{\parskip}{2mm}
\setlength{\textfloatsep}{5mm}
\setlength{\floatsep}{5mm}
\setlength{\intextsep}{5mm}
\setcounter{topnumber}{2}
\renewcommand{\topfraction}{.95}
\renewcommand{\textfraction}{.04}
\renewcommand{\floatpagefraction}{.7}
\emergencystretch=2em
\pagestyle{plain}
'''
        preamble += r'\hypersetup{pdftitle={'+escape_text(title)+r'},pdfauthor={MedVision Bench}}'+'\n'+r'\begin{document}'+'\n'
        tex_path.write_text(preamble+'\n\n'.join(self.tex)+'\n'+r'\end{document}'+'\n')
        engine = shutil.which('xelatex')
        if not engine:
            raise RuntimeError('XeLaTeX is required to render the manuscript equations')
        build_dir = self.root/'results/report_preview/latex_build'
        build_dir.mkdir(parents=True, exist_ok=True)
        command = [engine, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error',
                   '-file-line-error', '-output-directory='+build_dir.relative_to(self.root).as_posix(),
                   tex_path.relative_to(self.root).as_posix()]
        passes = []
        for _ in range(3):
            result = subprocess.run(command, cwd=self.root, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=90)
            passes.append(result.stdout)
            if result.returncode:
                (self.root/'results/latex_compile.log').write_text('\n'.join(passes))
                tail = '\n'.join(result.stdout.splitlines()[-40:])
                raise RuntimeError(
                    f'XeLaTeX failed (exit {result.returncode}); compiler output follows:\n'
                    f'{tail}\nFull output: results/latex_compile.log')
        log = (build_dir/'report.log').read_text(errors='replace')
        serious = [line for line in log.splitlines() if 'Missing character:' in line or
                   'Undefined control sequence' in line or 'There were undefined references' in line]
        if serious:
            raise RuntimeError('Typesetting audit failed: '+'; '.join(serious))
        shutil.copy2(build_dir/'report.pdf', output)
        (self.root/'results/latex_compile.log').write_text('\n'.join(passes))
        with pymupdf.open(output) as pdf:
            pages = len(pdf)
            for placement in self.placements:
                placement['pages'] = [i+1 for i,page in enumerate(pdf) if page.search_for(placement['label']+' |')]
        manifest = dict(engine='XeLaTeX', passes=3, pages=pages, native_display_equations=len(self.equations),
                        equation_labels=[row['label'] for row in self.equations], missing_characters=[],
                        undefined_controls=[], undefined_references=[],
                        overfull_boxes=re.findall(r'Overfull \\[hv]box.*', log),
                        source='report/report.tex', source_sha256=hashlib.sha256(tex_path.read_bytes()).hexdigest())
        (self.root/'report/typesetting_checks.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
