static void decode_p_block(FourXContext *f, uint16_t *dst, uint16_t *src,

                           int log2w, int log2h, int stride)

{

    const int index = size2index[log2h][log2w];

    const int h     = 1 << log2h;

    int code        = get_vlc2(&f->gb,

                               block_type_vlc[1 - (f->version > 1)][index].table,

                               BLOCK_TYPE_VLC_BITS, 1);

    uint16_t *start = (uint16_t *)f->last_picture.data[0];

    uint16_t *end   = start + stride * (f->avctx->height - h + 1) - (1 << log2w);



    av_assert2(code >= 0 && code <= 6);



    if (code == 0) {

        if (f->g.buffer_end - f->g.buffer < 1) {

            av_log(f->avctx, AV_LOG_ERROR, "bytestream overread\n");

            return;

        }

        src += f->mv[bytestream2_get_byte(&f->g)];

        if (start > src || src > end) {

            av_log(f->avctx, AV_LOG_ERROR, "mv out of pic\n");

            return;

        }

        mcdc(dst, src, log2w, h, stride, 1, 0);

    } else if (code == 1) {

        log2h--;

        decode_p_block(f, dst, src, log2w, log2h, stride);

        decode_p_block(f, dst + (stride << log2h),

                          src + (stride << log2h), log2w, log2h, stride);

    } else if (code == 2) {

        log2w--;

        decode_p_block(f, dst , src, log2w, log2h, stride);

        decode_p_block(f, dst + (1 << log2w),

                          src + (1 << log2w), log2w, log2h, stride);

    } else if (code == 3 && f->version < 2) {

        mcdc(dst, src, log2w, h, stride, 1, 0);

    } else if (code == 4) {

        if (f->g.buffer_end - f->g.buffer < 1) {

            av_log(f->avctx, AV_LOG_ERROR, "bytestream overread\n");

            return;

        }

        src += f->mv[bytestream2_get_byte(&f->g)];

        if (start > src || src > end) {

            av_log(f->avctx, AV_LOG_ERROR, "mv out of pic\n");

            return;

        }

        if (f->g2.buffer_end - f->g2.buffer < 1){

            av_log(f->avctx, AV_LOG_ERROR, "wordstream overread\n");

            return;

        }

        mcdc(dst, src, log2w, h, stride, 1, bytestream2_get_le16(&f->g2));

    } else if (code == 5) {

        if (f->g2.buffer_end - f->g2.buffer < 1) {

            av_log(f->avctx, AV_LOG_ERROR, "wordstream overread\n");

            return;

        }

        mcdc(dst, src, log2w, h, stride, 0, bytestream2_get_le16(&f->g2));

    } else if (code == 6) {

        if (f->g2.buffer_end - f->g2.buffer < 2) {

            av_log(f->avctx, AV_LOG_ERROR, "wordstream overread\n");

            return;

        }

        if (log2w) {

            dst[0]      = bytestream2_get_le16(&f->g2);

            dst[1]      = bytestream2_get_le16(&f->g2);

        } else {

            dst[0]      = bytestream2_get_le16(&f->g2);

            dst[stride] = bytestream2_get_le16(&f->g2);

        }

    }

}
