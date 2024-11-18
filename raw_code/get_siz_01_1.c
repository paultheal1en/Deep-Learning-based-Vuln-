static int get_siz(J2kDecoderContext *s)
{
    int i, ret;
    if (s->buf_end - s->buf < 36)
                        bytestream_get_be16(&s->buf); // Rsiz (skipped)
             s->width = bytestream_get_be32(&s->buf); // width
            s->height = bytestream_get_be32(&s->buf); // height
    s->image_offset_x = bytestream_get_be32(&s->buf); // X0Siz
    s->image_offset_y = bytestream_get_be32(&s->buf); // Y0Siz
        s->tile_width = bytestream_get_be32(&s->buf); // XTSiz
       s->tile_height = bytestream_get_be32(&s->buf); // YTSiz
     s->tile_offset_x = bytestream_get_be32(&s->buf); // XT0Siz
     s->tile_offset_y = bytestream_get_be32(&s->buf); // YT0Siz
       s->ncomponents = bytestream_get_be16(&s->buf); // CSiz
    if (s->buf_end - s->buf < 2 * s->ncomponents)
    for (i = 0; i < s->ncomponents; i++){ // Ssiz_i XRsiz_i, YRsiz_i
        uint8_t x = bytestream_get_byte(&s->buf);
        s->cbps[i] = (x & 0x7f) + 1;
        s->precision = FFMAX(s->cbps[i], s->precision);
        s->sgnd[i] = !!(x & 0x80);
        s->cdx[i] = bytestream_get_byte(&s->buf);
        s->cdy[i] = bytestream_get_byte(&s->buf);
    }
    s->numXtiles = ff_j2k_ceildiv(s->width - s->tile_offset_x, s->tile_width);
    s->numYtiles = ff_j2k_ceildiv(s->height - s->tile_offset_y, s->tile_height);
    s->tile = av_mallocz(s->numXtiles * s->numYtiles * sizeof(J2kTile));
    if (!s->tile)
        return AVERROR(ENOMEM);
    for (i = 0; i < s->numXtiles * s->numYtiles; i++){
        J2kTile *tile = s->tile + i;
        tile->comp = av_mallocz(s->ncomponents * sizeof(J2kComponent));
        if (!tile->comp)
            return AVERROR(ENOMEM);
    }
    s->avctx->width = s->width - s->image_offset_x;
    s->avctx->height = s->height - s->image_offset_y;
    switch(s->ncomponents){
        case 1: if (s->precision > 8) {
                    s->avctx->pix_fmt    = PIX_FMT_GRAY16;
                } else s->avctx->pix_fmt = PIX_FMT_GRAY8;
                break;
        case 3: if (s->precision > 8) {
                    s->avctx->pix_fmt    = PIX_FMT_RGB48;
                } else s->avctx->pix_fmt = PIX_FMT_RGB24;
                break;
        case 4: s->avctx->pix_fmt = PIX_FMT_BGRA; break;
    }
    if (s->picture.data[0])
        s->avctx->release_buffer(s->avctx, &s->picture);
    if ((ret = s->avctx->get_buffer(s->avctx, &s->picture)) < 0)
        return ret;
    s->picture.pict_type = FF_I_TYPE;
    s->picture.key_frame = 1;
    return 0;
}