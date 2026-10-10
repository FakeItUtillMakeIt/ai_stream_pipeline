// utils/http_util.h
// gate / report 节点共用的极小 HTTP + base64 工具，基于 libcurl。
// 只做 POST(JSON) 与读图转 base64，够 VLM 网关与告警上报用；不做连接池/异步。
#pragma once

#include <curl/curl.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <cstdlib>

namespace ai_stream {
namespace utils {
namespace http {

// curl 全局初始化只需一次，且必须线程安全。
inline void ensureGlobalInit()
{
    static std::once_flag flag;
    std::call_once(flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct Response
{
    long status = 0;
    std::string body;
    std::string err;    // 非空表示 curl 层失败（超时/连接等），status 不可信
    bool ok() const { return err.empty() && status >= 200 && status < 300; }
};

inline size_t writeCallback(void* ptr, size_t size, size_t nmemb, void* userdata)
{
    const size_t bytes = size * nmemb;
    static_cast<std::string*>(userdata)->append(static_cast<char*>(ptr), bytes);
    return bytes;
}

// POST 一个 JSON body。headers 形如 {"Authorization":"Bearer x"}。
// connect_timeout_ms 控制建连，timeout_ms 控制整体。
inline Response postJson(const std::string& url, const std::string& json_body,
                         const std::vector<std::string>& extra_headers,
                         long timeout_ms, long connect_timeout_ms = 3000)
{
    ensureGlobalInit();
    Response resp;
    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.err = "curl_easy_init failed";
        return resp;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    for (const auto& h : extra_headers)
        headers = curl_slist_append(headers, h.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(json_body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        resp.err = curl_easy_strerror(code);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

// 标准 base64（OpenAI image_url data url 用）。
inline std::string base64Encode(const std::vector<unsigned char>& in)
{
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        uint32_t n = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out += tbl[(n >> 18) & 0x3F];
        out += tbl[(n >> 12) & 0x3F];
        out += tbl[(n >> 6) & 0x3F];
        out += tbl[n & 0x3F];
        i += 3;
    }
    if (i < in.size()) {
        uint32_t n = in[i] << 16;
        if (i + 1 < in.size())
            n |= in[i + 1] << 8;
        out += tbl[(n >> 18) & 0x3F];
        out += tbl[(n >> 12) & 0x3F];
        out += (i + 1 < in.size()) ? tbl[(n >> 6) & 0x3F] : '=';
        out += '=';
    }
    return out;
}

// ${VAR} 展开：值形如 "${NAME}" 时用环境变量替换，方便将来把明文 key 切 env。
inline std::string expandEnv(const std::string& v)
{
    if (v.size() >= 3 && v.front() == '$' && v[1] == '{' && v.back() == '}') {
        const std::string name = v.substr(2, v.size() - 3);
        if (const char* val = std::getenv(name.c_str()))
            return val;
    }
    return v;
}

}  // namespace http
}  // namespace utils
}  // namespace ai_stream