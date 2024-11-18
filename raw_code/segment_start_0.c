static int segment_start(AVFormatContext *s, int write_header)

{

    SegmentContext *c = s->priv_data;

    AVFormatContext *oc = c->avf;

    int err = 0;



    if (write_header) {

        avformat_free_context(oc);

        c->avf = NULL;

        if ((err = segment_mux_init(s)) < 0)

            return err;

        oc = c->avf;

    }



    if (c->wrap)

        c->number %= c->wrap;



    if (av_get_frame_filename(oc->filename, sizeof(oc->filename),

                              s->filename, c->number++) < 0)

        return AVERROR(EINVAL);



    if ((err = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                          &s->interrupt_callback, NULL)) < 0)

        return err;



    if (oc->oformat->priv_class && oc->priv_data)

        av_opt_set(oc->priv_data, "resend_headers", "1", 0); /* mpegts specific */



    if (write_header) {

        if ((err = avformat_write_header(oc, NULL)) < 0)

            return err;

    }



    return 0;

}
