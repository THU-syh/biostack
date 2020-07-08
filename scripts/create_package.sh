cd packages
read -p 'package name: ' pname

if [ ! -f pname ]; then
    mkdir ${pname}
fi

cd ${pname}
mkdir tests

type nul > tests/__init__.py
type nul > tests/conftest.py
type nul > tests/test_predict.py

type nul > config.yml
type nul > MANIFEST.in
type nul > requirements.txt
type nul > setup.py

mkdir ${pname}
cd ${pname}

mkdir config
mkdir data
mkdir misc
mkdir predictions

type nul > config/__init__.py
type nul > config/config.py

type nul > misc/__init__.py
type nul > misc/callbacks.py
type nul > misc/data_management.py
type nul > misc/preprocessors.py

cd ..

type nul > ${pname}/__init__.py
type nul > ${pname}/model.py
type nul > ${pname}/pipeline.py
type nul > ${pname}/predict.py
type nul > ${pname}/train_pipeline.py
type nul > ${pname}/VERSION
