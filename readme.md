[HTTP Server] ---(POST JSON)---> [Pipeline Manager]
                                       |
                                       v
[RTSP Source Node] --(Raw H264)--> [Decode Node] --(YUV/RGB)--> [Queue/Broadcast]
                                                                      |
                    +-------------------------------------------------+-----------------+
                    |                                                 |                 |
                    v                                                 v                 v
           [Infer Node A]                                     [Infer Node B]    [Preview/Save]
           (Detection)                                        (Classification)  (Draw Raw)
                    |                                                 |
                    v                                                 v
           [Draw Node A]                                       [Draw Node B]
                    |                                                 |
                    +-----------------------+-------------------------+
                                            |
                                            v
                                   [Encode/RTMP Sink]