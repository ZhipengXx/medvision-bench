"""Regression checks for the manuscript's actual math-rendering failures."""
import shutil
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.report_typesetting import LatexManuscript, escape_text


def test_math_is_preserved_while_prose_characters_are_escaped(tmp_path):
    manuscript = LatexManuscript(tmp_path/'results', tmp_path, {9:'Grad-CAM'})
    md, tex = manuscript.format(r'LR 10^-4; $w_c$ and $\gamma=2$; 70% & AUROC [0,1]. [9]')
    assert '$10^{-4}$' in tex
    assert '$w_c$' in tex and r'$\gamma=2$' in tex
    assert r'70\% \&' in tex
    assert '[0,1]' in tex and r'\textsuperscript{1}' in tex
    assert r'\textbackslash' not in tex
    assert manuscript.citation_order == {9:1}


def test_legacy_math_exponents_do_not_create_nested_delimiters(tmp_path):
    manuscript = LatexManuscript(tmp_path/'results', tmp_path, {})
    _, tex = manuscript.format('$10^-4$ and 10^-3')
    assert tex == '$10^{-4}$ and $10^{-3}$'
    with pytest.raises(ValueError, match='Unbalanced'):
        escape_text('The symbol $gamma was not closed')


@pytest.mark.skipif(shutil.which('xelatex') is None, reason='XeLaTeX is not installed')
def test_equations_compile_as_native_searchable_math(tmp_path):
    manuscript = LatexManuscript(tmp_path/'results', tmp_path, {1:'Example reference'})
    manuscript.para('Formula rendering verification', 'Title')
    manuscript.para(r'Learning rates $10^{-4}$ and $10^{-3}$; $\gamma=2$, $n_c$ and accuracy 70%. [1]')
    manuscript.equation(
        r'\mathcal{L}_{\mathrm{WCE}}=-\frac{\sum_{i=1}^{m}w_{y_i}\log p_{i,y_i}}{\sum_{i=1}^{m}w_{y_i}}',
        'eq:weighted-loss', r'The denominator uses $w_{y_i}$.')
    manuscript.table('Table 1', 'Literal names and percent units.', ['Model','Accuracy'], [['ResNet_50','70.00%']])
    manuscript.references()
    output = tmp_path/'report/report.pdf'
    checks = manuscript.build(output, 'Typesetting verification')
    assert checks['native_display_equations'] == 1
    assert not checks['missing_characters'] and not checks['undefined_controls']
    with pymupdf.open(output) as pdf:
        text = '\n'.join(page.get_text() for page in pdf)
        assert '70%' in text and 'ResNet_50' in text
        assert not any(token in text for token in (r'\frac', r'\sum', '$', '10^-4'))
        fonts = [font[3] for page in pdf for font in page.get_fonts()]
        assert any('LatinModernMath' in font.replace('-', '') for font in fonts)
