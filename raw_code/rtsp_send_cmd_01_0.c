static void rtsp_send_cmd (AVFormatContext *s,

                           const char *cmd, RTSPMessageHeader *reply,

                           unsigned char **content_ptr)

{

    rtsp_send_cmd_async(s, cmd, reply, content_ptr);



    rtsp_read_reply(s, reply, content_ptr, 0);

}
