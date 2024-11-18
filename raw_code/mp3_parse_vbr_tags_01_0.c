static void mp3_parse_vbr_tags(AVFormatContext *s, AVStream *st, int64_t base)

{

    uint32_t v, spf;

    int frames = -1; /* Total number of frames in file */

    const int64_t xing_offtbl[2][2] = {{32, 17}, {17,9}};

    MPADecodeContext c;



    v = get_be32(s->pb);

    if(ff_mpa_check_header(v) < 0)

      return;



    ff_mpegaudio_decode_header(&c, v);

    if(c.layer != 3)

        return;



    /* Check for Xing / Info tag */

    url_fseek(s->pb, xing_offtbl[c.lsf == 1][c.nb_channels == 1], SEEK_CUR);

    v = get_be32(s->pb);

    if(v == MKBETAG('X', 'i', 'n', 'g') || v == MKBETAG('I', 'n', 'f', 'o')) {

        v = get_be32(s->pb);

        if(v & 0x1)

            frames = get_be32(s->pb);

    }



    /* Check for VBRI tag (always 32 bytes after end of mpegaudio header) */

    url_fseek(s->pb, base + 4 + 32, SEEK_SET);

    v = get_be32(s->pb);

    if(v == MKBETAG('V', 'B', 'R', 'I')) {

        /* Check tag version */

        if(get_be16(s->pb) == 1) {

            /* skip delay, quality and total bytes */

            url_fseek(s->pb, 8, SEEK_CUR);

            frames = get_be32(s->pb);

        }

    }



    if(frames < 0)

        return;



    spf = c.lsf ? 576 : 1152; /* Samples per frame, layer 3 */

    st->duration = av_rescale_q(frames, (AVRational){spf, c.sample_rate},

                                st->time_base);

}
