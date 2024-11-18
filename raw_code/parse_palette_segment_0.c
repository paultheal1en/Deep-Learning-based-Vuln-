static void parse_palette_segment(AVCodecContext *avctx,

                                  const uint8_t *buf, int buf_size)

{

    PGSSubContext *ctx = avctx->priv_data;



    const uint8_t *buf_end = buf + buf_size;

    const uint8_t *cm      = ff_crop_tab + MAX_NEG_CROP;

    int color_id;

    int y, cb, cr, alpha;

    int r, g, b, r_add, g_add, b_add;



    /* Skip two null bytes */

    buf += 2;



    while (buf < buf_end) {

        color_id  = bytestream_get_byte(&buf);

        y         = bytestream_get_byte(&buf);

        cr        = bytestream_get_byte(&buf);

        cb        = bytestream_get_byte(&buf);

        alpha     = bytestream_get_byte(&buf);



        YUV_TO_RGB1(cb, cr);

        YUV_TO_RGB2(r, g, b, y);



        av_dlog(avctx, "Color %d := (%d,%d,%d,%d)\n", color_id, r, g, b, alpha);



        /* Store color in palette */

        ctx->clut[color_id] = RGBA(r,g,b,alpha);

    }

}
