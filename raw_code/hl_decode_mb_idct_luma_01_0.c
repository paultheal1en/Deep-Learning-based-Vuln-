static av_always_inline void hl_decode_mb_idct_luma(const H264Context *h, H264SliceContext *sl,

                                                    int mb_type, int simple,

                                                    int transform_bypass,

                                                    int pixel_shift,

                                                    const int *block_offset,

                                                    int linesize,

                                                    uint8_t *dest_y, int p)

{

    void (*idct_add)(uint8_t *dst, int16_t *block, int stride);

    int i;

    block_offset += 16 * p;

    if (!IS_INTRA4x4(mb_type)) {

        if (IS_INTRA16x16(mb_type)) {

            if (transform_bypass) {

                if (h->sps.profile_idc == 244 &&

                    (sl->intra16x16_pred_mode == VERT_PRED8x8 ||

                     sl->intra16x16_pred_mode == HOR_PRED8x8)) {

                    h->hpc.pred16x16_add[sl->intra16x16_pred_mode](dest_y, block_offset,

                                                                   sl->mb + (p * 256 << pixel_shift),

                                                                   linesize);

                } else {

                    for (i = 0; i < 16; i++)

                        if (sl->non_zero_count_cache[scan8[i + p * 16]] ||

                            dctcoef_get(sl->mb, pixel_shift, i * 16 + p * 256))

                            h->h264dsp.h264_add_pixels4_clear(dest_y + block_offset[i],

                                                              sl->mb + (i * 16 + p * 256 << pixel_shift),

                                                              linesize);

                }

            } else {

                h->h264dsp.h264_idct_add16intra(dest_y, block_offset,

                                                sl->mb + (p * 256 << pixel_shift),

                                                linesize,

                                                sl->non_zero_count_cache + p * 5 * 8);

            }

        } else if (sl->cbp & 15) {

            if (transform_bypass) {

                const int di = IS_8x8DCT(mb_type) ? 4 : 1;

                idct_add = IS_8x8DCT(mb_type) ? h->h264dsp.h264_add_pixels8_clear

                    : h->h264dsp.h264_add_pixels4_clear;

                for (i = 0; i < 16; i += di)

                    if (sl->non_zero_count_cache[scan8[i + p * 16]])

                        idct_add(dest_y + block_offset[i],

                                 sl->mb + (i * 16 + p * 256 << pixel_shift),

                                 linesize);

            } else {

                if (IS_8x8DCT(mb_type))

                    h->h264dsp.h264_idct8_add4(dest_y, block_offset,

                                               sl->mb + (p * 256 << pixel_shift),

                                               linesize,

                                               sl->non_zero_count_cache + p * 5 * 8);

                else

                    h->h264dsp.h264_idct_add16(dest_y, block_offset,

                                               sl->mb + (p * 256 << pixel_shift),

                                               linesize,

                                               sl->non_zero_count_cache + p * 5 * 8);

            }

        }

    }

}
