import re
from pylatexenc.latex2text import LatexNodes2Text

# LaTeX Cleaning Functions
def clean_latex(latex_str):
    latex_str = latex_str.replace('\\\\', '\\')
    latex_str = re.sub(r"\\\[|\\\]", "", latex_str)
    latex_str = re.sub(r"\$+", "", latex_str)
    latex_str = latex_str.replace("```text", "").replace("```", "")
    latex_str = latex_str.replace("`text", "").replace("`", "")
    return latex_str

def latex_to_text(latex_str):
    cleaned = clean_latex(latex_str)
    return LatexNodes2Text().latex_to_text(cleaned)