static int mp3_write_audio_packet(AVFormatContext *s, AVPacket *pkt)

{

    MP3Context  *mp3 = s->priv_data;



    if (pkt->data && pkt->size >= 4) {

        MPADecodeHeader c;

        int av_unused base;



        avpriv_mpegaudio_decode_header(&c, AV_RB32(pkt->data));



        if (!mp3->initial_bitrate)

            mp3->initial_bitrate = c.bit_rate;

        if ((c.bit_rate == 0) || (mp3->initial_bitrate != c.bit_rate))

            mp3->has_variable_bitrate = 1;



#ifdef FILTER_VBR_HEADERS

        /* filter out XING and INFO headers. */

        base = 4 + xing_offtbl[c.lsf == 1][c.nb_channels == 1];



        if (base + 4 <= pkt->size) {

            uint32_t v = AV_RB32(pkt->data + base);



            if (MKBETAG('X','i','n','g') == v || MKBETAG('I','n','f','o') == v)

                return 0;

        }



        /* filter out VBRI headers. */

        base = 4 + 32;



        if (base + 4 <= pkt->size && MKBETAG('V','B','R','I') == AV_RB32(pkt->data + base))

            return 0;

#endif



        if (mp3->xing_offset)

            mp3_xing_add_frame(mp3, pkt);

    }



    return ff_raw_write_packet(s, pkt);

}
