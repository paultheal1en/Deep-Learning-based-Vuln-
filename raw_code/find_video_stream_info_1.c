static int find_video_stream_info(AVFormatContext *fmt_ctx, int decode)
{
    int ret = 0;
    int i, done = 0;
    AVPacket pkt;
    av_init_packet(&pkt);
    while (!done) {
        AVCodecContext *codec_ctx = NULL;
        AVStream *st;
        if ((ret = av_read_frame(fmt_ctx, &pkt)) < 0) {
            av_log(fmt_ctx, AV_LOG_ERROR, "Failed to read frame\n");
            goto end;
        st = fmt_ctx->streams[pkt.stream_index];
        codec_ctx = st->codec;
        /* Writing to AVStream.codec_info_nb_frames must not be done by
         * user applications. It is done here for testing purposing as
         * find_video_stream_info tries to mimic avformat_find_stream_info
         * which writes to this field.
         * */
        if (codec_ctx->codec_type != AVMEDIA_TYPE_VIDEO ||
            st->codec_info_nb_frames++ > 0) {
            av_packet_unref(&pkt);
            continue;
        ret = try_decode_video_frame(codec_ctx, &pkt, decode);
        if (ret < 0) {
            av_log(fmt_ctx, AV_LOG_ERROR, "Failed to decode video frame\n");
            goto end;
        av_packet_unref(&pkt);
        /* check if all video streams have demuxed a packet */
        done = 1;
            st = fmt_ctx->streams[i];
            codec_ctx = st->codec;
            if (codec_ctx->codec_type != AVMEDIA_TYPE_VIDEO)
                continue;
            done &= st->codec_info_nb_frames > 0;
end:
    av_packet_unref(&pkt);
    return ret < 0;