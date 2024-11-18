static int dvvideo_encode_frame(AVCodecContext *c, AVPacket *pkt,

                                const AVFrame *frame, int *got_packet)

{

    DVVideoContext *s = c->priv_data;

    int ret;



    s->sys = avpriv_dv_codec_profile(c);

    if (!s->sys || ff_dv_init_dynamic_tables(s->sys))

        return -1;

    if ((ret = ff_alloc_packet(pkt, s->sys->frame_size)) < 0) {

        av_log(c, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }



    c->pix_fmt                = s->sys->pix_fmt;

    s->frame                  = frame;

    c->coded_frame->key_frame = 1;

    c->coded_frame->pict_type = AV_PICTURE_TYPE_I;



    s->buf = pkt->data;

    c->execute(c, dv_encode_video_segment, s->sys->work_chunks, NULL,

               dv_work_pool_size(s->sys), sizeof(DVwork_chunk));



    emms_c();



    dv_format_frame(s, pkt->data);



    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
