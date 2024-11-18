static int ffm_write_packet(AVFormatContext *s, int stream_index,

                            UINT8 *buf, int size, int force_pts)

{

    AVStream *st = s->streams[stream_index];

    FFMStream *fst = st->priv_data;

    INT64 pts;

    UINT8 header[FRAME_HEADER_SIZE];

    int duration;



    if (st->codec.codec_type == CODEC_TYPE_AUDIO) {

        duration = ((float)st->codec.frame_size / st->codec.sample_rate * 1000000.0);

    } else {

        duration = (1000000.0 * FRAME_RATE_BASE / (float)st->codec.frame_rate);

    }



    pts = fst->pts;

    /* packet size & key_frame */

    header[0] = stream_index;

    header[1] = 0;

    if (st->codec.coded_picture->key_frame)

        header[1] |= FLAG_KEY_FRAME;

    header[2] = (size >> 16) & 0xff;

    header[3] = (size >> 8) & 0xff;

    header[4] = size & 0xff;

    header[5] = (duration >> 16) & 0xff;

    header[6] = (duration >> 8) & 0xff;

    header[7] = duration & 0xff;

    ffm_write_data(s, header, FRAME_HEADER_SIZE, pts, 1);

    ffm_write_data(s, buf, size, pts, 0);



    fst->pts += duration;

    return 0;

}
