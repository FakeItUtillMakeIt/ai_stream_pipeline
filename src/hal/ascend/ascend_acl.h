// src/hal/ascend/ascend_acl.h
// Ascend ACL 进程级初始化辅助（检测推理 / DVPP 解码共用）
#pragma once

#include <acl/acl.h>

#include <mutex>

namespace ai_stream {
namespace hal {

// aclInit 只能成功执行一次；多个 HAL 后端共用同一个 once_flag。
// 若 aclInit 返回“重复初始化”但设备可查询，也视为已就绪。
inline bool ensureAscendAclInitialized() {
    static std::once_flag flag;
    static aclError result = ACL_ERROR_NONE;
    std::call_once(flag, [] {
        aclError r = aclInit(nullptr);
        if (r != ACL_SUCCESS) {
            uint32_t count = 0;
            if (aclrtGetDeviceCount(&count) == ACL_SUCCESS && count > 0) {
                r = ACL_SUCCESS;
            }
        }
        result = r;
    });
    return result == ACL_SUCCESS;
}

} // namespace hal
} // namespace ai_stream
