# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list

from adversarialnlp.dataset_readers import ActivityNetCaptionsDatasetReader
from adversarialnlp.tests.utils import FIXTURES_ROOT

class TestActivityNetCaptionsReader():
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = ActivityNetCaptionsDatasetReader(lazy=lazy)
        instances = reader.read(FIXTURES_ROOT / 'activitynet_captions.json')
        instances = ensure_list(instances)

        instance1 = {"video_id": "v_uqiMw7tQ1Cc",
                     "first_sentence": "A weight lifting tutorial is given .".split(),
                     "second_sentence": "The coach helps the guy in red with the proper body placement and lifting technique .".split()}

        instance2 = {"video_id": "v_bXdq2zI1Ms0",
                     "first_sentence": "A man is seen speaking to the camera and pans out into more men standing behind him .".split(),
                     "second_sentence": "The first man then begins performing martial arts moves while speaking to he camera .".split()}

        instance3 = {"video_id": "v_bXdq2zI1Ms0",
                     "first_sentence": "The first man then begins performing martial arts moves while speaking to he camera .".split(),
                     "second_sentence": "He continues moving around and looking to the camera .".split()}

        assert len(instances) == 3

        for instance, expected_instance in zip(instances, [instance1, instance2, instance3]):
            fields = instance.fields
            assert [t.text for t in fields["first_sentence"].tokens] == expected_instance["first_sentence"]
            assert [t.text for t in fields["second_sentence"].tokens] == expected_instance["second_sentence"]
            assert fields["video_id"].metadata == expected_instance["video_id"]
