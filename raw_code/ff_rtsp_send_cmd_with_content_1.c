int ff_rtsp_send_cmd_with_content(AVFormatContext *s,

                                  const char *method, const char *url,

                                  const char *header,

                                  RTSPMessageHeader *reply,

                                  unsigned char **content_ptr,

                                  const unsigned char *send_content,

                                  int send_content_length)

{

    RTSPState *rt = s->priv_data;

    HTTPAuthType cur_auth_type;

    int ret;



retry:

    cur_auth_type = rt->auth_state.auth_type;

    if ((ret = ff_rtsp_send_cmd_with_content_async(s, method, url, header,

                                                   send_content,

                                                   send_content_length)))

        return ret;



    if ((ret = ff_rtsp_read_reply(s, reply, content_ptr, 0, method) ) < 0)

        return ret;



    if (reply->status_code == 401 && cur_auth_type == HTTP_AUTH_NONE &&

        rt->auth_state.auth_type != HTTP_AUTH_NONE)

        goto retry;



    if (reply->status_code > 400){

        av_log(s, AV_LOG_ERROR, "method %s failed: %d%s\n",

               method,

               reply->status_code,

               reply->reason);

        av_log(s, AV_LOG_DEBUG, "%s\n", rt->last_reply);

    }



    return 0;

}
