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
mkdir utilities
mkdir predictions

type nul > config/__init__.py
type nul > config/config.py

type nul > utilities/__init__.py
type nul > utilities/callbacks.py
type nul > utilities/data_management.py
type nul > utilities/preprocessors.py

cd ..

type nul > ${pname}/__init__.py
type nul > ${pname}/model.py
type nul > ${pname}/pipeline.py
type nul > ${pname}/predict.py
type nul > ${pname}/train_pipeline.py
type nul > ${pname}/VERSION
