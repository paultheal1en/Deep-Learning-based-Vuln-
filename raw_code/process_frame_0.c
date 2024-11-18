static av_always_inline int process_frame(WriterContext *w,

                                          InputFile *ifile,

                                          AVFrame *frame, AVPacket *pkt)

{

    AVFormatContext *fmt_ctx = ifile->fmt_ctx;

    AVCodecContext *dec_ctx = ifile->streams[pkt->stream_index].dec_ctx;

    AVCodecParameters *par = ifile->streams[pkt->stream_index].st->codecpar;

    AVSubtitle sub;

    int ret = 0, got_frame = 0;



    if (dec_ctx->codec) {

        switch (par->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            ret = avcodec_decode_video2(dec_ctx, frame, &got_frame, pkt);

            break;



        case AVMEDIA_TYPE_AUDIO:

            ret = avcodec_decode_audio4(dec_ctx, frame, &got_frame, pkt);

            break;



        case AVMEDIA_TYPE_SUBTITLE:

            ret = avcodec_decode_subtitle2(dec_ctx, &sub, &got_frame, pkt);

            break;

        }

    }



    if (ret < 0)

        return ret;

    ret = FFMIN(ret, pkt->size); /* guard against bogus return values */

    pkt->data += ret;

    pkt->size -= ret;

    if (got_frame) {

        int is_sub = (par->codec_type == AVMEDIA_TYPE_SUBTITLE);

        nb_streams_frames[pkt->stream_index]++;

        if (do_show_frames)

            if (is_sub)

                show_subtitle(w, &sub, ifile->streams[pkt->stream_index].st, fmt_ctx);

            else

                show_frame(w, frame, ifile->streams[pkt->stream_index].st, fmt_ctx);

        if (is_sub)

            avsubtitle_free(&sub);

    }

    return got_frame;

}
