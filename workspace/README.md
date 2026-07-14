# workspace/

This is your folder. Build the bot here, in any language, with any deps.

The mock environment lives one directory up; you should not need to edit
anything outside `workspace/`. Treat `mock/` as an external system you're
integrating against — same way you'd treat the real Discord API in
production.

## What you need to know

1. Your bot must accept HTTP POSTs at `http://localhost:8080/interactions`
   (default; override via `BOT_WEBHOOK_URL` in `.env`).
2. The full API contract for the three mock services is in
   [`../docs/PROTOCOL.md`](../docs/PROTOCOL.md).
3. The product spec / business context is in
   [`../README.md`](../README.md).
4. When something feels wrong: [`../docs/HINTS.md`](../docs/HINTS.md).
5. Design decisions & edge-case handling in
   [`处理思路.md`](./处理思路.md).

---

## Quick start

### 1. 启动 mock 服务（Docker）

```bash
# 在项目根目录执行
cd ..
make dev
```

### 2. 启动 bot

**方式 A — 宿主机直接运行**

```bash
pip install -r requirements.txt
python bot.py
```

**方式 B — Docker 运行**

```bash
# 构建镜像
docker build -t opus-bot .

# 启动容器（连接宿主机上的 mock 服务）
docker run -d --name opus-bot \
  -p 8080:8080 \
  -e MOCK_DISCORD_URL=http://host.docker.internal:7001 \
  -e MOCK_PLATFORM_URL=http://host.docker.internal:7002 \
  -e MOCK_LLM_URL=http://host.docker.internal:7003 \
  -e CALLBACK_HOST=http://host.docker.internal:8080 \
  opus-bot

# 查看日志
docker logs -f opus-bot

# 停止并删除
docker stop opus-bot && docker rm opus-bot
```

### 3. 触发命令

```bash
# 开另一个终端
cd ..
make logs                     # 观察频道消息
make send url=https://...     # 触发 /opus 命令
```

### 4. 运行单元测试

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

预期输出：`29 passed in 0.26s`

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `bot.py` | 主程序 — 全部三个 Phase 的实现 |
| `Dockerfile` | Docker 镜像构建 |
| `requirements.txt` | Python 依赖 |
| `pytest.ini` | pytest 配置（asyncio_mode = auto） |
| `tests/test_bot.py` | 29 个单元测试 |
| `处理思路.md` | 异步接入设计思路、代码组织、边界处理说明 |
| `README.md` | 本文件 |

---

## 测试覆盖

| 测试类 | 数量 | 覆盖内容 |
|--------|------|---------|
| `TestInteractions` | 6 | ping/pong、/opus deferred ACK、参数校验、未知命令 |
| `TestTokenAPI` | 5 | 创建/领取/重复/过期/无效 token |
| `TestTranscript` | 3 | 字幕获取成功/HTTP错误/超时 |
| `TestLLMWindow` | 4 | JSON解析/500降级/超时降级/散文提取 |
| `TestClipCallback` | 5 | 幂等性/状态校验/成功流程/重复回调 |
| `TestStorage` | 5 | 原子存储、弹出、认领操作 |
| `TestHealth` | 1 | 健康检查 |

---

## 处理思路

详细的设计决策、异步流程、边界处理策略请见 [`处理思路.md`](./处理思路.md)，包括：

- Discord 3 秒超时约束的应对
- Pipeline 异步延迟的处理（deferred + webhook 回调）
- LLM 对抗性输出的三层防御
- 幂等性设计（重复回调、重复领取）
- 网络异常处理与系统代理绕过
- 完整端到端数据流时序图
