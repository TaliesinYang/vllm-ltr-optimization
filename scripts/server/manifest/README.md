# 固定版本与 OSS 清单

- `gateway-pin.txt` 是 VeloxMesh fork 的不可移动 commit SHA。
- `oss-objects.json` 是 restore 的唯一对象清单，记录 URI、原始 bytes SHA-256、真实字节数、归档格式与预期解包内容。

提交态允许 `size_bytes`/`sha256`/`unpacks_to` 为 `null`，表示 Mac repack 尚未实际执行且归档真实成员路径尚未记录。`inventory_and_repack.sh` 会从归档目录表提取 concrete member paths，并在计算真值后原子覆盖这些字段；`restore_from_oss.sh` 拒绝任何未填、零值、空串或非 64 位 SHA 的对象，并对同尺寸的已有缓存仍重新计算 SHA。
