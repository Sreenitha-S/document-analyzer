# interface2.spec
block_cipher = None

a = Analysis(
    ['interface2.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.SentenceTransformer',
        'transformers',
        'transformers.pipelines',
        'transformers.tokenization_utils_base',
        'transformers.models.auto',
        'transformers.models.bert',
        'transformers.modeling_utils',
        'tokenizers',
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.utils.data',
        'numpy',
        'charset_normalizer.legacy'
    ],
    excludes=[
        'tensorboard',
        'torch.utils.tensorboard'
    ],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='interface2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='interface2'
)
