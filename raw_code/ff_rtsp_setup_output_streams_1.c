int ff_rtsp_setup_output_streams(AVFormatContext *s, const char *addr)

{

    RTSPState *rt = s->priv_data;

    RTSPMessageHeader reply1, *reply = &reply1;

    int i;

    char *sdp;

    AVFormatContext sdp_ctx, *ctx_array[1];



    s->start_time_realtime = av_gettime();



    /* Announce the stream */

    sdp = av_mallocz(SDP_MAX_SIZE);

    if (sdp == NULL)

        return AVERROR(ENOMEM);

    /* We create the SDP based on the RTSP AVFormatContext where we

     * aren't allowed to change the filename field. (We create the SDP

     * based on the RTSP context since the contexts for the RTP streams

     * don't exist yet.) In order to specify a custom URL with the actual

     * peer IP instead of the originally specified hostname, we create

     * a temporary copy of the AVFormatContext, where the custom URL is set.

     *

     * FIXME: Create the SDP without copying the AVFormatContext.

     * This either requires setting up the RTP stream AVFormatContexts

     * already here (complicating things immensely) or getting a more

     * flexible SDP creation interface.

     */

    sdp_ctx = *s;

    ff_url_join(sdp_ctx.filename, sizeof(sdp_ctx.filename),

                "rtsp", NULL, addr, -1, NULL);

    ctx_array[0] = &sdp_ctx;

    if (avf_sdp_create(ctx_array, 1, sdp, SDP_MAX_SIZE)) {

        av_free(sdp);

        return AVERROR_INVALIDDATA;

    }

    av_log(s, AV_LOG_VERBOSE, "SDP:\n%s\n", sdp);

    ff_rtsp_send_cmd_with_content(s, "ANNOUNCE", rt->control_uri,

                                  "Content-Type: application/sdp\r\n",

                                  reply, NULL, sdp, strlen(sdp));

    av_free(sdp);

    if (reply->status_code != RTSP_STATUS_OK)

        return AVERROR_INVALIDDATA;



    /* Set up the RTSPStreams for each AVStream */

    for (i = 0; i < s->nb_streams; i++) {

        RTSPStream *rtsp_st;

        AVStream *st = s->streams[i];



        rtsp_st = av_mallocz(sizeof(RTSPStream));

        if (!rtsp_st)

            return AVERROR(ENOMEM);

        dynarray_add(&rt->rtsp_streams, &rt->nb_rtsp_streams, rtsp_st);



        st->priv_data = rtsp_st;

        rtsp_st->stream_index = i;



        av_strlcpy(rtsp_st->control_url, rt->control_uri, sizeof(rtsp_st->control_url));

        /* Note, this must match the relative uri set in the sdp content */

        av_strlcatf(rtsp_st->control_url, sizeof(rtsp_st->control_url),

                    "/streamid=%d", i);

    }



    return 0;

}
