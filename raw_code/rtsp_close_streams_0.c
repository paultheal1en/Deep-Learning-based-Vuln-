static void rtsp_close_streams(RTSPState *rt)

{

    int i;

    RTSPStream *rtsp_st;



    for(i=0;i<rt->nb_rtsp_streams;i++) {

        rtsp_st = rt->rtsp_streams[i];

        if (rtsp_st) {

            if (rtsp_st->transport_priv) {

                if (rt->transport == RTSP_TRANSPORT_RDT)

                    ff_rdt_parse_close(rtsp_st->transport_priv);

                else

                    rtp_parse_close(rtsp_st->transport_priv);

            }

            if (rtsp_st->rtp_handle)

                url_close(rtsp_st->rtp_handle);

            if (rtsp_st->dynamic_handler && rtsp_st->dynamic_protocol_context)

                rtsp_st->dynamic_handler->close(rtsp_st->dynamic_protocol_context);

        }

    }

    av_free(rt->rtsp_streams);

    if (rt->asf_ctx) {

        av_close_input_stream (rt->asf_ctx);

        rt->asf_ctx = NULL;

    }

    av_freep(&rt->auth_b64);

}
