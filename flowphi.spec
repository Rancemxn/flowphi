from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Blender runs these modules with its own Python, so ship their source too.
datas = collect_data_files('flowphi', include_py_files=True)
datas += [('README.md', '.'), ('LICENSE', '.')]
for package in ('glcontext', 'numpy', 'pyside6', 'rich', 'shapely', 'skia-python'):
    datas += copy_metadata(package, recursive=True)

a = Analysis(
    ['main.py'],
    pathex=[SPECPATH],
    datas=datas,
    hiddenimports=collect_submodules('glcontext'),
    excludes=['bpy', 'bpy_extras', 'mathutils', 'tkinter'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='FlowPhi', console=False, upx=False)
COLLECT(exe, a.binaries, a.datas, name='FlowPhi', upx=False)
