static int segment_start(AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    int err = 0;



    if (seg->wrap)

        seg->number %= seg->wrap;



    if (av_get_frame_filename(oc->filename, sizeof(oc->filename),

                              s->filename, seg->number++) < 0)

        return AVERROR(EINVAL);



    if ((err = avio_open2(&oc->pb, oc->filename, AVIO_FLAG_WRITE,

                          &s->interrupt_callback, NULL)) < 0)

        return err;



    if (!oc->priv_data && oc->oformat->priv_data_size > 0) {

        oc->priv_data = av_mallocz(oc->oformat->priv_data_size);

        if (!oc->priv_data) {

            avio_close(oc->pb);

            return AVERROR(ENOMEM);

        }

        if (oc->oformat->priv_class) {

            *(const AVClass**)oc->priv_data = oc->oformat->priv_class;

            av_opt_set_defaults(oc->priv_data);

        }

    }



    if ((err = oc->oformat->write_header(oc)) < 0) {

        goto fail;

    }



    return 0;



fail:

    av_log(oc, AV_LOG_ERROR, "Failure occurred when starting segment '%s'\n",

           oc->filename);

    avio_close(oc->pb);

    av_freep(&oc->priv_data);



    return err;

}
