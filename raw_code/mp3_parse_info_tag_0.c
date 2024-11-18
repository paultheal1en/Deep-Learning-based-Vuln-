static void mp3_parse_info_tag(AVFormatContext *s, AVStream *st,

                               MPADecodeHeader *c, uint32_t spf)

{

#define LAST_BITS(k, n) ((k) & ((1 << (n)) - 1))

#define MIDDLE_BITS(k, m, n) LAST_BITS((k) >> (m), ((n) - (m)))



    uint16_t crc;

    uint32_t v;



    char version[10];



    uint32_t peak   = 0;

    int32_t  r_gain = INT32_MIN, a_gain = INT32_MIN;



    MP3DecContext *mp3 = s->priv_data;

    static const int64_t xing_offtbl[2][2] = {{32, 17}, {17,9}};

    uint64_t fsize = avio_size(s->pb);



    /* Check for Xing / Info tag */

    avio_skip(s->pb, xing_offtbl[c->lsf == 1][c->nb_channels == 1]);

    v = avio_rb32(s->pb);

    mp3->is_cbr = v == MKBETAG('I', 'n', 'f', 'o');

    if (v != MKBETAG('X', 'i', 'n', 'g') && !mp3->is_cbr)

        return;



    v = avio_rb32(s->pb);

    if (v & XING_FLAG_FRAMES)

        mp3->frames = avio_rb32(s->pb);

    if (v & XING_FLAG_SIZE)

        mp3->header_filesize = avio_rb32(s->pb);

    if (fsize && mp3->header_filesize) {

        uint64_t min, delta;

        min = FFMIN(fsize, mp3->header_filesize);

        delta = FFMAX(fsize, mp3->header_filesize) - min;

        if (fsize > mp3->header_filesize && delta > min >> 4) {

            mp3->frames = 0;

        } else if (delta > min >> 4) {

            av_log(s, AV_LOG_WARNING,

                   "filesize and duration do not match (growing file?)\n");

        }

    }

    if (v & XING_FLAG_TOC)

        read_xing_toc(s, mp3->header_filesize, av_rescale_q(mp3->frames,

                                       (AVRational){spf, c->sample_rate},

                                       st->time_base));

    /* VBR quality */

    if(v & 8)

        avio_skip(s->pb, 4);



    /* Encoder short version string */

    memset(version, 0, sizeof(version));

    avio_read(s->pb, version, 9);



    /* Info Tag revision + VBR method */

    avio_r8(s->pb);



    /* Lowpass filter value */

    avio_r8(s->pb);



    /* ReplayGain peak */

    v    = avio_rb32(s->pb);

    peak = av_rescale(v, 100000, 1 << 23);



    /* Radio ReplayGain */

    v = avio_rb16(s->pb);



    if (MIDDLE_BITS(v, 13, 15) == 1) {

        r_gain = MIDDLE_BITS(v, 0, 8) * 10000;



        if (v & (1 << 9))

            r_gain *= -1;

    }



    /* Audiophile ReplayGain */

    v = avio_rb16(s->pb);



    if (MIDDLE_BITS(v, 13, 15) == 2) {

        a_gain = MIDDLE_BITS(v, 0, 8) * 10000;



        if (v & (1 << 9))

            a_gain *= -1;

    }



    /* Encoding flags + ATH Type */

    avio_r8(s->pb);



    /* if ABR {specified bitrate} else {minimal bitrate} */

    avio_r8(s->pb);



    /* Encoder delays */

    v= avio_rb24(s->pb);

    if(AV_RB32(version) == MKBETAG('L', 'A', 'M', 'E')

        || AV_RB32(version) == MKBETAG('L', 'a', 'v', 'f')) {



        mp3->start_pad = v>>12;

        mp3->  end_pad = v&4095;

        st->skip_samples = mp3->start_pad + 528 + 1;

        if (mp3->frames)

            st->end_discard_sample = -mp3->end_pad + 528 + 1 + mp3->frames * (int64_t)spf;

        if (!st->start_time)

            st->start_time = av_rescale_q(st->skip_samples,

                                            (AVRational){1, c->sample_rate},

                                            st->time_base);

        av_log(s, AV_LOG_DEBUG, "pad %d %d\n", mp3->start_pad, mp3->  end_pad);

    }



    /* Misc */

    avio_r8(s->pb);



    /* MP3 gain */

    avio_r8(s->pb);



    /* Preset and surround info */

    avio_rb16(s->pb);



    /* Music length */

    avio_rb32(s->pb);



    /* Music CRC */

    avio_rb16(s->pb);



    /* Info Tag CRC */

    crc = ffio_get_checksum(s->pb);

    v   = avio_rb16(s->pb);



    if (v == crc) {

        ff_replaygain_export_raw(st, r_gain, peak, a_gain, 0);

        av_dict_set(&st->metadata, "encoder", version, 0);

    }

}
