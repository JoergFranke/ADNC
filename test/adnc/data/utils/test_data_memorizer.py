# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pytest
import shutil
import pathlib
from adnc.data.utils.data_memorizer import DataMemorizer

TMP_DIR = '.tmp_dir'


@pytest.fixture()
def tmp_dir():
    tmp_dir = TMP_DIR
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir)


class TestDataMemorizer():
    def test_hashing(self, tmp_dir):
        hash_config_1 = {'set_types': 'tokens', 'target_mode': 'mode1', 'seed': 123}
        hash_config_2 = {'seed': 123, 'set_types': 'tokens', 'target_mode': 'mode1'}
        hash_config_3 = {'set_types': 'tokens', 'target_mode': 'mode1', 'seed': 124}

        data_memory_1 = DataMemorizer(hash_config_1, tmp_dir)
        data_memory_2 = DataMemorizer(hash_config_2, tmp_dir)
        data_memory_3 = DataMemorizer(hash_config_3, tmp_dir)

        assert data_memory_1.hash_name == data_memory_2.hash_name
        assert data_memory_1.hash_name != data_memory_3.hash_name

    def test_data_memorizing(self, tmp_dir):
        hash_config = {'set_types': 'tokens', 'target_mode': 'mode1', 'seed': 123}
        dummy_data = [{'dict': 'test'}, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'string_test']

        data_memory = DataMemorizer(hash_config, tmp_dir)
        assert not data_memory()

        data_memory.dump_data(dummy_data)
        assert data_memory()

        dict_dummy, list_dummy, str_dummy = data_memory.load_data()
        assert dict_dummy['dict'] == 'test'
        assert list_dummy[4] == 4
        assert str_dummy == 'string_test'

        data_memory.purge_data()
        assert not data_memory()
