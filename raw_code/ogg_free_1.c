static void ogg_free(AVFormatContext *s)

{

    int i;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        OGGStreamContext *oggstream = st->priv_data;



        if (st->codecpar->codec_id == AV_CODEC_ID_FLAC ||

            st->codecpar->codec_id == AV_CODEC_ID_SPEEX ||

            st->codecpar->codec_id == AV_CODEC_ID_OPUS ||

            st->codecpar->codec_id == AV_CODEC_ID_VP8) {

            av_freep(&oggstream->header[0]);

        }

        av_freep(&oggstream->header[1]);

        av_freep(&st->priv_data);

    }

}