static int transcode_subtitles(InputStream *ist, AVPacket *pkt, int *got_output)

{

    AVSubtitle subtitle;

    int i, ret = avcodec_decode_subtitle2(ist->dec_ctx,

                                          &subtitle, got_output, pkt);



    check_decode_result(got_output, ret);



    if (ret < 0 || !*got_output) {

        if (!pkt->size)

            sub2video_flush(ist);

        return ret;

    }



    if (ist->fix_sub_duration) {

        int end = 1;

        if (ist->prev_sub.got_output) {

            end = av_rescale(subtitle.pts - ist->prev_sub.subtitle.pts,

                             1000, AV_TIME_BASE);

            if (end < ist->prev_sub.subtitle.end_display_time) {

                av_log(ist->dec_ctx, AV_LOG_DEBUG,

                       "Subtitle duration reduced from %d to %d%s\n",

                       ist->prev_sub.subtitle.end_display_time, end,

                       end <= 0 ? ", dropping it" : "");

                ist->prev_sub.subtitle.end_display_time = end;

            }

        }

        FFSWAP(int,        *got_output, ist->prev_sub.got_output);

        FFSWAP(int,        ret,         ist->prev_sub.ret);

        FFSWAP(AVSubtitle, subtitle,    ist->prev_sub.subtitle);

        if (end <= 0)

            goto out;

    }



    if (!*got_output)

        return ret;



    sub2video_update(ist, &subtitle);



    if (!subtitle.num_rects)

        goto out;



    ist->frames_decoded++;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        if (!check_output_constraints(ist, ost) || !ost->encoding_needed

            || ost->enc->type != AVMEDIA_TYPE_SUBTITLE)

            continue;



        do_subtitle_out(output_files[ost->file_index]->ctx, ost, ist, &subtitle);

    }



out:

    avsubtitle_free(&subtitle);

    return ret;

}
