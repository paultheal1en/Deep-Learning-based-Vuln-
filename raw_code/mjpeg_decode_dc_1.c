static inline int mjpeg_decode_dc(MJpegDecodeContext *s, int dc_index)

{

    int code;

    code = get_vlc2(&s->gb, s->vlcs[0][dc_index].table, 9, 2);

    if (code < 0) {

        av_log(s->avctx, AV_LOG_WARNING,

               "mjpeg_decode_dc: bad vlc: %d:%d (%p)\n",

               0, dc_index, &s->vlcs[0][dc_index]);

        return 0xffff;

    }



    if (code)

        return get_xbits(&s->gb, code);

    else

        return 0;

}
