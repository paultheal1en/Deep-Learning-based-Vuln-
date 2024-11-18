static uint8_t get_sot(Jpeg2000DecoderContext *s, int n)

{

    Jpeg2000TilePart *tp;

    uint16_t Isot;

    uint32_t Psot;

    uint8_t TPsot;



    if (s->buf_end - s->buf < 4)

        return AVERROR(EINVAL);



    Isot = bytestream_get_be16(&s->buf);        // Isot

    if (Isot) {

        av_log(s->avctx, AV_LOG_ERROR,

               "Not a DCINEMA JP2K file: more than one tile\n");

        return -1;

    }

    Psot  = bytestream_get_be32(&s->buf);       // Psot

    TPsot = bytestream_get_byte(&s->buf);       // TPsot



    /* Read TNSot but not used */

    bytestream_get_byte(&s->buf);               // TNsot



    tp             = s->tile[s->curtileno].tile_part + TPsot;

    tp->tile_index = Isot;

    tp->tp_len     = Psot;

    tp->tp_idx     = TPsot;



    /* Start of bit stream. Pointer to SOD marker

     * Check SOD marker is present. */

    if (JPEG2000_SOD == bytestream_get_be16(&s->buf))

        tp->tp_start_bstrm = s->buf;

    else {

        av_log(s->avctx, AV_LOG_ERROR, "SOD marker not found \n");

        return -1;

    }



    /* End address of bit stream =

     *     start address + (Psot - size of SOT HEADER(n)

     *     - size of SOT MARKER(2)  - size of SOD marker(2) */

    tp->tp_end_bstrm = s->buf + (tp->tp_len - n - 4);



    // set buffer pointer to end of tile part header

    s->buf = tp->tp_end_bstrm;



    return 0;

}
