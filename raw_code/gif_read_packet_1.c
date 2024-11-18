static int gif_read_packet(AVFormatContext * s1,

                           AVPacket * pkt)

{

    GifState *s = s1->priv_data;

    int ret;



    ret = gif_parse_next_image(s);

    if (ret < 0)

        return ret;



    /* XXX: avoid copying */

    if (av_new_packet(pkt, s->screen_width * s->screen_height * 3)) {

        return AVERROR(EIO);

    }

    pkt->stream_index = 0;

    memcpy(pkt->data, s->image_buf, s->screen_width * s->screen_height * 3);

    return 0;

}
