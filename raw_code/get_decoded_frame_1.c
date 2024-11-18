static av_always_inline int get_decoded_frame(AVFormatContext *fmt_ctx,

                                              AVFrame *frame, int *got_frame,

                                              AVPacket *pkt)

{

    AVCodecContext *dec_ctx = fmt_ctx->streams[pkt->stream_index]->codec;

    int ret = 0;



    *got_frame = 0;

    if (dec_ctx->codec) {

    switch (dec_ctx->codec_type) {

    case AVMEDIA_TYPE_VIDEO:

        ret = avcodec_decode_video2(dec_ctx, frame, got_frame, pkt);

        break;



    case AVMEDIA_TYPE_AUDIO:

        ret = avcodec_decode_audio4(dec_ctx, frame, got_frame, pkt);

        break;





    return ret;
