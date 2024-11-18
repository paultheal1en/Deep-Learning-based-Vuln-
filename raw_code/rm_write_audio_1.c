static int rm_write_audio(AVFormatContext *s, const uint8_t *buf, int size, int flags)

{

    uint8_t *buf1;

    RMMuxContext *rm = s->priv_data;

    AVIOContext *pb = s->pb;

    StreamInfo *stream = rm->audio_stream;

    int i;



    /* XXX: suppress this malloc */

    buf1 = av_malloc(size * sizeof(uint8_t));





    write_packet_header(s, stream, size, !!(flags & AV_PKT_FLAG_KEY));



    if (stream->enc->codec_id == AV_CODEC_ID_AC3) {

        /* for AC-3, the words seem to be reversed */

        for(i=0;i<size;i+=2) {

            buf1[i] = buf[i+1];

            buf1[i+1] = buf[i];

        }

        avio_write(pb, buf1, size);

    } else {

        avio_write(pb, buf, size);

    }

    stream->nb_frames++;

    av_free(buf1);

    return 0;

}