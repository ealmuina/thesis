# -*- mode: python -*-

block_cipher = None


a = Analysis(['clusterapp/__main__.py'],
             pathex=['/home/eddy/PycharmProjects/thesis'],
             binaries=[],
             datas=[
                ('clusterapp/templates', 'templates'),
                ('clusterapp/static', 'static')
             ],
             hiddenimports=[
                'scipy._lib.messagestream',
                'sklearn.neighbors.typedefs',
                'sklearn.neighbors.quad_tree',
                'sklearn.tree',
                'sklearn.tree._utils'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='__main__',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='__main__')
