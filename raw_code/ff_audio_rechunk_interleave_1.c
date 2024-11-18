int ff_audio_rechunk_interleave(AVFormatContext *s, AVPacket *out, AVPacket *pkt, int flush,

                        int (*get_packet)(AVFormatContext *, AVPacket *, AVPacket *, int),

                        int (*compare_ts)(AVFormatContext *, AVPacket *, AVPacket *))

{

    int i;



    if (pkt) {

        AVStream *st = s->streams[pkt->stream_index];

        AudioInterleaveContext *aic = st->priv_data;

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            unsigned new_size = av_fifo_size(aic->fifo) + pkt->size;

            if (new_size > aic->fifo_size) {

                if (av_fifo_realloc2(aic->fifo, new_size) < 0)

                    return -1;

                aic->fifo_size = new_size;

            }

            av_fifo_generic_write(aic->fifo, pkt->data, pkt->size, NULL);

        } else {

            // rewrite pts and dts to be decoded time line position

            pkt->pts = pkt->dts = aic->dts;

            aic->dts += pkt->duration;

            ff_interleave_add_packet(s, pkt, compare_ts);

        }

        pkt = NULL;

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            AVPacket new_pkt;

            while (ff_interleave_new_audio_packet(s, &new_pkt, i, flush))

                ff_interleave_add_packet(s, &new_pkt, compare_ts);

        }

    }



    return get_packet(s, out, pkt, flush);

}
