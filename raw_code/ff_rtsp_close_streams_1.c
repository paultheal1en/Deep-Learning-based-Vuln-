void ff_rtsp_close_streams(AVFormatContext *s)

{

    RTSPState *rt = s->priv_data;

    int i;

    RTSPStream *rtsp_st;



    ff_rtsp_undo_setup(s);

    for (i = 0; i < rt->nb_rtsp_streams; i++) {

        rtsp_st = rt->rtsp_streams[i];

        if (rtsp_st) {

            if (rtsp_st->dynamic_handler && rtsp_st->dynamic_protocol_context)

                rtsp_st->dynamic_handler->close(

                    rtsp_st->dynamic_protocol_context);


        }

    }

    av_free(rt->rtsp_streams);

    if (rt->asf_ctx) {

        av_close_input_stream (rt->asf_ctx);

        rt->asf_ctx = NULL;

    }

    av_free(rt->p);

    av_free(rt->recvbuf);

}