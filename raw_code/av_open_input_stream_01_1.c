int av_open_input_stream(AVFormatContext **ic_ptr,

                         AVIOContext *pb, const char *filename,

                         AVInputFormat *fmt, AVFormatParameters *ap)

{

    int err;

    AVDictionary *opts;

    AVFormatContext *ic;

    AVFormatParameters default_ap;



    if(!ap){

        ap=&default_ap;

        memset(ap, 0, sizeof(default_ap));

    }

    opts = convert_format_parameters(ap);



    if(!ap->prealloced_context)

        ic = avformat_alloc_context();

    else

        ic = *ic_ptr;

    if (!ic) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    if (pb && fmt && fmt->flags & AVFMT_NOFILE)

        av_log(ic, AV_LOG_WARNING, "Custom AVIOContext makes no sense and "

                                   "will be ignored with AVFMT_NOFILE format.\n");

    else

        ic->pb = pb;



    err = avformat_open_input(&ic, filename, fmt, &opts);

    ic->pb = ic->pb ? ic->pb : pb; // don't leak custom pb if it wasn't set above



    *ic_ptr = ic;

fail:

    av_dict_free(&opts);

    return err;

}
