static int transcode_subtitles(InputStream *ist, AVPacket *pkt, int *got_output)

{

    AVSubtitle subtitle;

    int i, ret = avcodec_decode_subtitle2(ist->dec_ctx,

                                          &subtitle, got_output, pkt);

    if (ret < 0)

        return ret;

    if (!*got_output)

        return ret;



    ist->frames_decoded++;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        if (!check_output_constraints(ist, ost) || !ost->encoding_needed)

            continue;



        do_subtitle_out(output_files[ost->file_index]->ctx, ost, ist, &subtitle, pkt->pts);

    }



    avsubtitle_free(&subtitle);

    return ret;

}
