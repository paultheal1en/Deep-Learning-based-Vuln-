static int get_sot(Jpeg2000DecoderContext *s, int n)
{
    Jpeg2000TilePart *tp;
    uint16_t Isot;
    uint32_t Psot;
    uint8_t TPsot;
    if (bytestream2_get_bytes_left(&s->g) < 8)
    Isot = bytestream2_get_be16u(&s->g);        // Isot
    if (Isot) {
        avpriv_request_sample(s->avctx, "Support for more than one tile");
        return AVERROR_PATCHWELCOME;
    }
    Psot  = bytestream2_get_be32u(&s->g);       // Psot
    TPsot = bytestream2_get_byteu(&s->g);       // TPsot
    /* Read TNSot but not used */
    bytestream2_get_byteu(&s->g);               // TNsot
    if (Psot > bytestream2_get_bytes_left(&s->g) + n + 2) {
        av_log(s->avctx, AV_LOG_ERROR, "Psot %d too big\n", Psot);
    }
    if (TPsot >= FF_ARRAY_ELEMS(s->tile[Isot].tile_part)) {
        avpriv_request_sample(s->avctx, "Support for %d components", TPsot);
        return AVERROR_PATCHWELCOME;
    }
    tp             = s->tile[s->curtileno].tile_part + TPsot;
    tp->tile_index = Isot;
    tp->tp_len     = Psot;
    tp->tp_idx     = TPsot;
    /* Start of bit stream. Pointer to SOD marker
     * Check SOD marker is present. */
    if (JPEG2000_SOD == bytestream2_get_be16(&s->g)) {
        bytestream2_init(&tp->tpg, s->g.buffer, tp->tp_len - n - 4);
        bytestream2_skip(&s->g, tp->tp_len - n - 4);
    } else {
        av_log(s->avctx, AV_LOG_ERROR, "SOD marker not found \n");
    }
    /* End address of bit stream =
     *     start address + (Psot - size of SOT HEADER(n)
     *     - size of SOT MARKER(2)  - size of SOD marker(2) */
    return 0;
}