static int gxf_interleave_packet(AVFormatContext *s, AVPacket *out, AVPacket *pkt, int flush)

{

    GXFContext *gxf = s->priv_data;

    AVPacket new_pkt;

    int i;



    for (i = 0; i < s->nb_streams; i++) {

        if (s->streams[i]->codec->codec_type == CODEC_TYPE_AUDIO) {

            GXFStreamContext *sc = &gxf->streams[i];

            if (pkt && pkt->stream_index == i) {

                av_fifo_write(&sc->audio_buffer, pkt->data, pkt->size);

                pkt = NULL;

            }

            if (flush || av_fifo_size(&sc->audio_buffer) >= GXF_AUDIO_PACKET_SIZE) {

                if (!pkt && gxf_new_audio_packet(gxf, sc, &new_pkt, flush) > 0) {

                    pkt = &new_pkt;

                    break; /* add pkt right now into list */

                }

            }

        }

    }

    return av_interleave_packet_per_dts(s, out, pkt, flush);

}
