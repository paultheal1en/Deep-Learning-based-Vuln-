static int decode_video(InputStream *ist, AVPacket *pkt, int *got_output)

{

    AVFrame *decoded_frame, *f;

    int i, ret = 0, err = 0;



    if (!ist->decoded_frame && !(ist->decoded_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    if (!ist->filter_frame && !(ist->filter_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    decoded_frame = ist->decoded_frame;



    ret = decode(ist->dec_ctx, decoded_frame, got_output, pkt);

    if (!*got_output || ret < 0)

        return ret;



    ist->frames_decoded++;



    if (ist->hwaccel_retrieve_data && decoded_frame->format == ist->hwaccel_pix_fmt) {

        err = ist->hwaccel_retrieve_data(ist->dec_ctx, decoded_frame);

        if (err < 0)

            goto fail;

    }

    ist->hwaccel_retrieved_pix_fmt = decoded_frame->format;



    decoded_frame->pts = guess_correct_pts(&ist->pts_ctx, decoded_frame->pts,

                                           decoded_frame->pkt_dts);

    if (ist->framerate.num)

        decoded_frame->pts = ist->cfr_next_pts++;



    if (ist->st->sample_aspect_ratio.num)

        decoded_frame->sample_aspect_ratio = ist->st->sample_aspect_ratio;



    for (i = 0; i < ist->nb_filters; i++) {

        if (i < ist->nb_filters - 1) {

            f = ist->filter_frame;

            err = av_frame_ref(f, decoded_frame);

            if (err < 0)

                break;

        } else

            f = decoded_frame;



        err = ifilter_send_frame(ist->filters[i], f);

        if (err < 0)

            break;

    }



fail:

    av_frame_unref(ist->filter_frame);

    av_frame_unref(decoded_frame);

    return err < 0 ? err : ret;

}
