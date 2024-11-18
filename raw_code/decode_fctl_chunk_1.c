static int decode_fctl_chunk(AVCodecContext *avctx, PNGDecContext *s,

                             uint32_t length)

{

    uint32_t sequence_number;

    int cur_w, cur_h, x_offset, y_offset, dispose_op, blend_op;



    if (length != 26)

        return AVERROR_INVALIDDATA;



    if (!(s->state & PNG_IHDR)) {

        av_log(avctx, AV_LOG_ERROR, "fctl before IHDR\n");

        return AVERROR_INVALIDDATA;

    }



    s->last_w = s->cur_w;

    s->last_h = s->cur_h;

    s->last_x_offset = s->x_offset;

    s->last_y_offset = s->y_offset;

    s->last_dispose_op = s->dispose_op;



    sequence_number = bytestream2_get_be32(&s->gb);

    cur_w           = bytestream2_get_be32(&s->gb);

    cur_h           = bytestream2_get_be32(&s->gb);

    x_offset        = bytestream2_get_be32(&s->gb);

    y_offset        = bytestream2_get_be32(&s->gb);

    bytestream2_skip(&s->gb, 4); /* delay_num (2), delay_den (2) */

    dispose_op      = bytestream2_get_byte(&s->gb);

    blend_op        = bytestream2_get_byte(&s->gb);

    bytestream2_skip(&s->gb, 4); /* crc */



    if (sequence_number == 0 &&

        (cur_w != s->width ||

         cur_h != s->height ||

         x_offset != 0 ||

         y_offset != 0) ||

        cur_w <= 0 || cur_h <= 0 ||

        x_offset < 0 || y_offset < 0 ||

        cur_w > s->width - x_offset|| cur_h > s->height - y_offset)

            return AVERROR_INVALIDDATA;



    if (blend_op != APNG_BLEND_OP_OVER && blend_op != APNG_BLEND_OP_SOURCE) {

        av_log(avctx, AV_LOG_ERROR, "Invalid blend_op %d\n", blend_op);

        return AVERROR_INVALIDDATA;

    }



    if ((sequence_number == 0 || !s->previous_picture.f->data[0]) &&

        dispose_op == APNG_DISPOSE_OP_PREVIOUS) {

        // No previous frame to revert to for the first frame

        // Spec says to just treat it as a APNG_DISPOSE_OP_BACKGROUND

        dispose_op = APNG_DISPOSE_OP_BACKGROUND;

    }



    if (blend_op == APNG_BLEND_OP_OVER && !s->has_trns && (

            avctx->pix_fmt == AV_PIX_FMT_RGB24 ||

            avctx->pix_fmt == AV_PIX_FMT_RGB48BE ||

            avctx->pix_fmt == AV_PIX_FMT_PAL8 ||

            avctx->pix_fmt == AV_PIX_FMT_GRAY8 ||

            avctx->pix_fmt == AV_PIX_FMT_GRAY16BE ||

            avctx->pix_fmt == AV_PIX_FMT_MONOBLACK

        )) {

        // APNG_BLEND_OP_OVER is the same as APNG_BLEND_OP_SOURCE when there is no alpha channel

        blend_op = APNG_BLEND_OP_SOURCE;

    }



    s->cur_w      = cur_w;

    s->cur_h      = cur_h;

    s->x_offset   = x_offset;

    s->y_offset   = y_offset;

    s->dispose_op = dispose_op;

    s->blend_op   = blend_op;



    return 0;

}
