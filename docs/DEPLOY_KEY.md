# GitHub Deploy Key 配置

目标仓库：`rollinpioneer/vp`

在 GitHub 仓库的 `Settings -> Deploy keys -> Add deploy key` 中填写：

- **Title**：`vico-point experiment host 2026-09-06`
- **Key**：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJdXibWpyYvXz31LyrjB+DIUhdYKT+Nt9/qae+jjaPIJ rollinpioneer/vp deploy key 2026-09-06
```

- **Allow write access**：必须勾选，否则只能拉取，不能推送。

本机私钥为 `~/.ssh/vp_github_deploy_ed25519`。私钥不得粘贴到 GitHub、提交到仓库或发送给他人。当前仓库远端使用 SSH 别名：

```text
git@github-vp:rollinpioneer/vp.git
```

公钥指纹：

```text
SHA256:6wuhThFnmAvdn4eNM+BBj1TmIUk4Q4PAjnw3Hq+zd9w
```
