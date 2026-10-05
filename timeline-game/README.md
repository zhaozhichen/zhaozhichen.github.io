# 计算帝国

一个人的历史排序游戏。输入名字，将未知年份的事件插入时间轴，揭晓年份并阅读历史背景。

[开始游戏](https://zhaozhichen.github.io/timeline-game/)

## 功能

- 284 张卡片、568 张独立正反面 PNG；每局不重复抽取。
- 相同年份的相邻位置均算正确，答错后自动归位，可继续挑战。
- 记录作答数、正误与正确率。初始卡和未回答的卡不计入正确率分母。
- 至少回答 10 张才可结束，之后可继续挑战。进度保存在当前浏览器，刷新可继续。
- 结束后生成含姓名、分数、结束时间和游戏二维码的 PNG 分享卡，没有排行榜。
- 历史卡附录支持搜索、主题筛选、正反面阅读和原作来源查看。
- 黑白响应式界面，支持键盘与手机横向时间轴。

## 开发

纯 HTML / CSS / JavaScript，无构建步骤。由仓库根目录的静态服务器访问 `/timeline-game/`。

运行规则测试：`node --test timeline-game/tests/game.test.cjs`。

主要文件：`index.html`、`style.css`、`app.js`、`core.js`、`data.js`、`share.js`、`qr.js`。卡牌图片与数据位于 `cards/`。

## 原作与使用说明

原作与原始图像来自 [Calculating Empires](https://calculatingempires.net/)，作者 Kate Crawford 与 Vladan Joler。部分图像经 AI 去除文字遮挡并补全；部分为同主题概念配图，具体说明见卡片详情。

本项目仅供个人学习与娱乐，不做任何商业用途，是非官方个人项目，与原作者及展览机构无隶属或合作关系。原作与原始图像的权利归相应权利人所有。仓库其他部分的许可说明不构成对这些原作及图像的额外授权。

年份口径和来源见 `cards/date-audit.json`，图像处理记录见 `cards/art-process.json`。

标题与玩家姓名使用思源宋体 Heavy（Adobe Source Han Serif SC Heavy）的自托管 WOFF2 子集，按 SIL OFL 1.1 授权；字体来源、修改说明及许可见 `fonts/README.md` 和 `fonts/OFL.txt`。
