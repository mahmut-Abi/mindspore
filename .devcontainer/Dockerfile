FROM uos-server-2500:latest

RUN echo -e '[koji-v25]\nname=koji-v25\nbaseurl=https://koji.uniontech.com/kojifiles/repos/V25-build/latest/$basearch/\nenabled=1\ngpgcheck=0\nmodule_hotfixes=1' > /etc/yum.repos.d/uos.repo
RUN dnf makecache && dnf install -y tar fish git

# COPY .devcontainer/welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
