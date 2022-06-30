# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

exitcode=0
for fname in `find . -name "*.py"`; 
do
  diff -q <(head -14 $fname) <(cat license-header.txt) > /dev/null
  if [ $? -ne 0 ]
  then
    echo "$fname does not have license header. Please copy and paste ./license-header.txt"
    exitcode=1
  fi
done

for fname in `find . -name "*.sh"`; 
do
  diff -q <(head -13 $fname) <(tail -13 license-header.txt) > /dev/null
  if [ $? -ne 0 ]
  then
    echo "$fname does not have license header. Please copy and paste ./license-header.txt"
    exitcode=1
  fi
done

exit ${exitcode}
