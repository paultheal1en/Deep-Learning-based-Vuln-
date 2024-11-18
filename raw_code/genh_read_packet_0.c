static int genh_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVCodecContext *codec = s->streams[0]->codec;

    GENHDemuxContext *c = s->priv_data;

    int ret;



    if (c->dsp_int_type == 1 && codec->codec_id == AV_CODEC_ID_ADPCM_THP &&

        codec->channels > 1) {

        int i, ch;



        if (avio_feof(s->pb))

            return AVERROR_EOF;

        av_new_packet(pkt, 8 * codec->channels);

        for (i = 0; i < 8 / c->interleave_size; i++) {

            for (ch = 0; ch < codec->channels; ch++) {

                pkt->data[ch * 8 + i*c->interleave_size+0] = avio_r8(s->pb);

                pkt->data[ch * 8 + i*c->interleave_size+1] = avio_r8(s->pb);

            }

        }

        ret = 0;

    } else {

        ret = av_get_packet(s->pb, pkt, codec->block_align ? codec->block_align : 1024 * codec->channels);

    }



    pkt->stream_index = 0;

    return ret;

}
