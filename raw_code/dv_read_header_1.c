static int dv_read_header(AVFormatContext *s,

                          AVFormatParameters *ap)

{

    unsigned state;

    RawDVContext *c = s->priv_data;



    c->dv_demux = dv_init_demux(s);

    if (!c->dv_demux)

        return -1;



    state = get_be32(s->pb);

    while ((state & 0xffffff7f) != 0x1f07003f) {

        if (url_feof(s->pb)) {

            av_log(s, AV_LOG_ERROR, "Cannot find DV header.\n");

            return -1;

        }

        state = (state << 8) | get_byte(s->pb);

    }

    AV_WB32(c->buf, state);



    if (get_buffer(s->pb, c->buf + 4, DV_PROFILE_BYTES - 4) <= 0 ||

        url_fseek(s->pb, -DV_PROFILE_BYTES, SEEK_CUR) < 0)

        return AVERROR(EIO);



    c->dv_demux->sys = dv_frame_profile(c->buf);

    if (!c->dv_demux->sys) {

        av_log(s, AV_LOG_ERROR, "Can't determine profile of DV input stream.\n");

        return -1;

    }



    s->bit_rate = av_rescale_q(c->dv_demux->sys->frame_size, (AVRational){8,1},

                               c->dv_demux->sys->time_base);



    return 0;

}
