# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG CLI_IMAGE
ARG TASK_IMAGE
FROM ${CLI_IMAGE} AS cli_runtime
RUN tar -C /usr/local --exclude=python3.12.13/lib/python3.12/site-packages \
    --exclude=python3.12.13/include --exclude=python3.12.13/share \
    -cf /tmp/hyper-python-runtime.tar python3.12.13

FROM ${TASK_IMAGE}
COPY --from=cli_runtime /opt/codex /opt/codex
COPY --from=cli_runtime /tmp/hyper-python-runtime.tar /tmp/hyper-python-runtime.tar
RUN tar -C /usr/local -xf /tmp/hyper-python-runtime.tar && rm /tmp/hyper-python-runtime.tar \
    && mkdir -p /workspace /opt/hyper-agent-home /opt/hyper-codex-home /etc/profile.d \
    && printf '%s\n' 'export PATH=/opt/codex/bin:/opt/miniconda3/envs/testbed/bin:$PATH' \
       > /etc/profile.d/hyper-codex.sh
# Official task images retain problem branches as remote-tracking references.
RUN git -C /testbed for-each-ref --format='%(refname:strip=3)' refs/remotes/origin \
    | while IFS= read -r branch; do \
        if [ "$branch" != HEAD ]; then \
            git -C /testbed show-ref --verify --quiet "refs/heads/$branch" \
            || git -C /testbed update-ref "refs/heads/$branch" "refs/remotes/origin/$branch" \
            || exit 1; \
        fi; \
      done
ENV PATH=/opt/codex/bin:/opt/miniconda3/envs/testbed/bin:${PATH} \
    HOME=/opt/hyper-agent-home CODEX_HOME=/opt/hyper-codex-home \
    PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/sleep", "infinity"]
