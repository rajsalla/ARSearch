import os
import glob
import re

def read_datys_data():
    """
        This funtion helps read data provided by DATYS
    """
    fqn_threads_dict = {}
    for _f in glob.glob("data/so_threads/*"):
        with open(_f, "r") as fp:
            if "notjava" in _f:
                continue
            thread_id = _f.split(os.sep)[-1].split(".")[0]
            content = fp.read()
            pattern = '<API label="(.*?)">(.*?)</API>'
            for line in content.split("\n"):
                match = re.search(pattern, line)
                while match is not None:
                    s = match.start()
                    matching_tag = match.group(0)
                    label =  match.group(1).strip()
                    api = match.group(2)
                    if label == 'com.google.common.collect.Sets.difference"':
                        label = 'com.google.common.collect.Sets.difference'
                    elif label == 'com.google.common.collect.EnumBiMap.pu':
                        label = 'com.google.common.collect.EnumBiMap.put'
                    elif label == 'org.mockito.stubbing.OngoingStubbing':
                        label = 'org.mockito.stubbing.OngoingStubbing.thenReturn'
                    elif label == 'com.google.common.collect.' and thread_id == '5716267':
                        label = 'com.google.common.collect.Multimaps.index'
                    elif label == 'com.google.common.io.Closer.reclose' and thread_id == '39658005':
                        label = 'com.google.common.io.Closer.close'
                    elif label == 'om.google.common.collect.BiMap' and thread_id == '61625556':
                        label = 'com.google.common.collect.BiMap.synchronizedBiMap'
                    elif label == "org.mockito.Mockito.argThat" and thread_id == "23273230":
                        label = ""
                    elif label == "org.assertj.core.api.OptionalIntAssert." and thread_id == "48866139":
                        label = ""
                    elif label == "org.mockito.Mockito.then" and thread_id == "42082918":
                        label = "org.mockito.Mockito.when"
                    elif label == "org.mockito.stubbing.OngoingStubbing.thenThrow" and thread_id == "19155369":
                        label = "org.mockito.stubbing.OngoingStubbing.thenReturn"
                    if label != "None" and label != "":
                        if label not in fqn_threads_dict:
                            fqn_threads_dict[label] = []
                        if thread_id not in fqn_threads_dict[label]:
                            fqn_threads_dict[label].append(thread_id)
                        
                    line = re.sub(re.escape(matching_tag), api, line, 1)
                    match = re.search(pattern, line)

            for k in ["c", "verify", "andDo", "Bibe", "Nome", "trimResults", "expireAfterWrite", "Ordering.natural", "Objects.equal"]:
                fqn_threads_dict.pop(k)
            pop_key_list = []

            for fqn, label in fqn_threads_dict.items():
                # do datys    
                if "</API>" in fqn:
                    pop_key_list.append(fqn)
            for k in pop_key_list:
                fqn_threads_dict.pop(k)
    return fqn_threads_dict