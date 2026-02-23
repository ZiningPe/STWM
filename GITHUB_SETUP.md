# GitHub 推送步骤与命令解释

## 每条命令的含义

```bash
cd ~/Downloads/stwm_package
```
**cd = Change Directory（切换目录）**
进入你存放包文件的文件夹。`~` 表示你的主目录（/Users/你的用户名）。
根据你实际解压的位置修改路径。

---

```bash
git init
```
**初始化 Git 仓库**
在当前文件夹创建隐藏的 `.git` 文件夹，让 Git 开始追踪所有文件变化。只需执行一次。

---

```bash
git add .
```
**把所有文件加入"暂存区"**
`.` 表示当前目录下的全部文件。就像打包行李——先把东西放进箱子，还没寄出去。

---

```bash
git commit -m "Initial commit: STWM package"
```
**正式保存一个版本快照**
`-m` 后面是这次提交的说明（版本备注）。就像把箱子封好并贴上标签。

---

```bash
git remote add origin https://github.com/ZiningPe/STWM.git
```
**告诉本地 Git：远程仓库在哪里**
`remote add` = 添加一个远程地址；`origin` = 这个地址的别名（约定俗成叫 origin）；后面的 URL = 你的 GitHub 仓库地址。

---

```bash
git branch -M main
```
**把当前分支重命名为 main**
GitHub 默认主分支叫 `main`，这条命令确保本地名称一致。

---

```bash
git push -u origin main
```
**把本地代码上传到 GitHub**
`push` = 推送/上传；`-u` = 设置默认上传目标（以后直接 `git push` 就够了）；`origin main` = 推送到 origin 的 main 分支。

---

## 你的完整操作流程（复制粘贴到 Terminal）

```bash
cd ~/Downloads/stwm_package   # 换成你的实际路径
git init
git add .
git commit -m "Initial commit: STWM package"
git remote add origin https://github.com/ZiningPe/STWM.git
git branch -M main
git push -u origin main
```

推送时会弹出登录窗口，输入 GitHub 用户名和密码。
如果提示需要 token：GitHub → Settings → Developer settings → Personal access tokens → 生成一个，粘贴替代密码。

---

## 以后每次更新代码只需三条

```bash
git add .
git commit -m "描述这次改了什么"
git push
```
