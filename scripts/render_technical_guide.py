"""Use the packaged DOCX renderer with installed Word as its PDF backend on Windows.

The packaged LibreOffice backend is unavailable on this Windows runtime.
No document edits are made by the read-only Word export.
"""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RUNTIME=Path('C:/Users/abdul/.cache/codex-runtimes/codex-primary-runtime/dependencies')
PACKAGED=Path('C:/Users/abdul/.codex/plugins/cache/openai-primary-runtime/documents/26.905.11957/skills/documents/render_docx.py')
spec=importlib.util.spec_from_file_location('packaged_renderer',PACKAGED)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
os.environ['PATH']=str(RUNTIME/'native'/'poppler'/'Library'/'bin')+os.pathsep+os.environ.get('PATH','')


def word_pdf(input_path,user_profile,out_dir,*args,**kwargs):
    target=Path(out_dir)/(Path(input_path).stem+'.pdf')
    proc=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'render_word.ps1'),
                         '-InputPath',str(Path(input_path).resolve()),'-OutputPath',str(target.resolve())],
                         capture_output=True,text=True,timeout=120)
    if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    if not target.is_file():raise RuntimeError('Word did not export the PDF')
    return str(target),proc.stdout+proc.stderr


renderer.convert_to_pdf=word_pdf
if __name__=='__main__':renderer.main()
