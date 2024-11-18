static av_noinline void FUNC(hl_decode_mb)(H264Context *h)

{

    const int mb_x    = h->mb_x;

    const int mb_y    = h->mb_y;

    const int mb_xy   = h->mb_xy;

    const int mb_type = h->cur_pic.f.mb_type[mb_xy];

    uint8_t *dest_y, *dest_cb, *dest_cr;

    int linesize, uvlinesize /*dct_offset*/;

    int i, j;

    int *block_offset = &h->block_offset[0];

    const int transform_bypass = !SIMPLE && (h->qscale == 0 && h->sps.transform_bypass);

    /* is_h264 should always be true if SVQ3 is disabled. */

    const int is_h264 = !CONFIG_SVQ3_DECODER || SIMPLE || h->avctx->codec_id == AV_CODEC_ID_H264;

    void (*idct_add)(uint8_t *dst, int16_t *block, int stride);

    const int block_h   = 16 >> h->chroma_y_shift;

    const int chroma422 = CHROMA422;



    dest_y  = h->cur_pic.f.data[0] + ((mb_x << PIXEL_SHIFT)     + mb_y * h->linesize)  * 16;

    dest_cb = h->cur_pic.f.data[1] +  (mb_x << PIXEL_SHIFT) * 8 + mb_y * h->uvlinesize * block_h;

    dest_cr = h->cur_pic.f.data[2] +  (mb_x << PIXEL_SHIFT) * 8 + mb_y * h->uvlinesize * block_h;



    h->vdsp.prefetch(dest_y  + (h->mb_x & 3) * 4 * h->linesize   + (64 << PIXEL_SHIFT), h->linesize,       4);

    h->vdsp.prefetch(dest_cb + (h->mb_x & 7)     * h->uvlinesize + (64 << PIXEL_SHIFT), dest_cr - dest_cb, 2);



    h->list_counts[mb_xy] = h->list_count;



    if (!SIMPLE && MB_FIELD) {

        linesize     = h->mb_linesize = h->linesize * 2;

        uvlinesize   = h->mb_uvlinesize = h->uvlinesize * 2;

        block_offset = &h->block_offset[48];

        if (mb_y & 1) { // FIXME move out of this function?

            dest_y  -= h->linesize * 15;

            dest_cb -= h->uvlinesize * (block_h - 1);

            dest_cr -= h->uvlinesize * (block_h - 1);

        }

        if (FRAME_MBAFF) {

            int list;

            for (list = 0; list < h->list_count; list++) {

                if (!USES_LIST(mb_type, list))

                    continue;

                if (IS_16X16(mb_type)) {

                    int8_t *ref = &h->ref_cache[list][scan8[0]];

                    fill_rectangle(ref, 4, 4, 8, (16 + *ref) ^ (h->mb_y & 1), 1);

                } else {

                    for (i = 0; i < 16; i += 4) {

                        int ref = h->ref_cache[list][scan8[i]];

                        if (ref >= 0)

                            fill_rectangle(&h->ref_cache[list][scan8[i]], 2, 2,

                                           8, (16 + ref) ^ (h->mb_y & 1), 1);

                    }

                }

            }

        }

    } else {

        linesize   = h->mb_linesize   = h->linesize;

        uvlinesize = h->mb_uvlinesize = h->uvlinesize;

        // dct_offset = s->linesize * 16;

    }



    if (!SIMPLE && IS_INTRA_PCM(mb_type)) {

        const int bit_depth = h->sps.bit_depth_luma;

        if (PIXEL_SHIFT) {

            int j;

            GetBitContext gb;

            init_get_bits(&gb, (uint8_t *)h->intra_pcm_ptr,

                          ff_h264_mb_sizes[h->sps.chroma_format_idc] * bit_depth);



            for (i = 0; i < 16; i++) {

                uint16_t *tmp_y = (uint16_t *)(dest_y + i * linesize);

                for (j = 0; j < 16; j++)

                    tmp_y[j] = get_bits(&gb, bit_depth);

            }

            if (SIMPLE || !CONFIG_GRAY || !(h->flags & CODEC_FLAG_GRAY)) {

                if (!h->sps.chroma_format_idc) {

                    for (i = 0; i < block_h; i++) {

                        uint16_t *tmp_cb = (uint16_t *)(dest_cb + i * uvlinesize);

                        uint16_t *tmp_cr = (uint16_t *)(dest_cr + i * uvlinesize);

                        for (j = 0; j < 8; j++) {

                            tmp_cb[j] = tmp_cr[j] = 1 << (bit_depth - 1);

                        }

                    }

                } else {

                    for (i = 0; i < block_h; i++) {

                        uint16_t *tmp_cb = (uint16_t *)(dest_cb + i * uvlinesize);

                        for (j = 0; j < 8; j++)

                            tmp_cb[j] = get_bits(&gb, bit_depth);

                    }

                    for (i = 0; i < block_h; i++) {

                        uint16_t *tmp_cr = (uint16_t *)(dest_cr + i * uvlinesize);

                        for (j = 0; j < 8; j++)

                            tmp_cr[j] = get_bits(&gb, bit_depth);

                    }

                }

            }

        } else {

            for (i = 0; i < 16; i++)

                memcpy(dest_y + i * linesize, (uint8_t *)h->intra_pcm_ptr + i * 16, 16);

            if (SIMPLE || !CONFIG_GRAY || !(h->flags & CODEC_FLAG_GRAY)) {

                if (!h->sps.chroma_format_idc) {

                    for (i = 0; i < 8; i++) {

                        memset(dest_cb + i*uvlinesize, 1 << (bit_depth - 1), 8);

                        memset(dest_cr + i*uvlinesize, 1 << (bit_depth - 1), 8);

                    }

                } else {

                    uint8_t *src_cb = (uint8_t *)h->intra_pcm_ptr + 256;

                    uint8_t *src_cr = (uint8_t *)h->intra_pcm_ptr + 256 + block_h * 8;

                    for (i = 0; i < block_h; i++) {

                        memcpy(dest_cb + i * uvlinesize, src_cb + i * 8, 8);

                        memcpy(dest_cr + i * uvlinesize, src_cr + i * 8, 8);

                    }

                }

            }

        }

    } else {

        if (IS_INTRA(mb_type)) {

            if (h->deblocking_filter)

                xchg_mb_border(h, dest_y, dest_cb, dest_cr, linesize,

                               uvlinesize, 1, 0, SIMPLE, PIXEL_SHIFT);



            if (SIMPLE || !CONFIG_GRAY || !(h->flags & CODEC_FLAG_GRAY)) {

                h->hpc.pred8x8[h->chroma_pred_mode](dest_cb, uvlinesize);

                h->hpc.pred8x8[h->chroma_pred_mode](dest_cr, uvlinesize);

            }



            hl_decode_mb_predict_luma(h, mb_type, is_h264, SIMPLE,

                                      transform_bypass, PIXEL_SHIFT,

                                      block_offset, linesize, dest_y, 0);



            if (h->deblocking_filter)

                xchg_mb_border(h, dest_y, dest_cb, dest_cr, linesize,

                               uvlinesize, 0, 0, SIMPLE, PIXEL_SHIFT);

        } else if (is_h264) {

            if (chroma422) {

                FUNC(hl_motion_422)(h, dest_y, dest_cb, dest_cr,

                              h->me.qpel_put, h->h264chroma.put_h264_chroma_pixels_tab,

                              h->me.qpel_avg, h->h264chroma.avg_h264_chroma_pixels_tab,

                              h->h264dsp.weight_h264_pixels_tab,

                              h->h264dsp.biweight_h264_pixels_tab);

            } else {

                FUNC(hl_motion_420)(h, dest_y, dest_cb, dest_cr,

                              h->me.qpel_put, h->h264chroma.put_h264_chroma_pixels_tab,

                              h->me.qpel_avg, h->h264chroma.avg_h264_chroma_pixels_tab,

                              h->h264dsp.weight_h264_pixels_tab,

                              h->h264dsp.biweight_h264_pixels_tab);

            }

        }



        hl_decode_mb_idct_luma(h, mb_type, is_h264, SIMPLE, transform_bypass,

                               PIXEL_SHIFT, block_offset, linesize, dest_y, 0);



        if ((SIMPLE || !CONFIG_GRAY || !(h->flags & CODEC_FLAG_GRAY)) &&

            (h->cbp & 0x30)) {

            uint8_t *dest[2] = { dest_cb, dest_cr };

            if (transform_bypass) {

                if (IS_INTRA(mb_type) && h->sps.profile_idc == 244 &&

                    (h->chroma_pred_mode == VERT_PRED8x8 ||

                     h->chroma_pred_mode == HOR_PRED8x8)) {

                    h->hpc.pred8x8_add[h->chroma_pred_mode](dest[0],

                                                            block_offset + 16,

                                                            h->mb + (16 * 16 * 1 << PIXEL_SHIFT),

                                                            uvlinesize);

                    h->hpc.pred8x8_add[h->chroma_pred_mode](dest[1],

                                                            block_offset + 32,

                                                            h->mb + (16 * 16 * 2 << PIXEL_SHIFT),

                                                            uvlinesize);

                } else {

                    idct_add = h->h264dsp.h264_add_pixels4;

                    for (j = 1; j < 3; j++) {

                        for (i = j * 16; i < j * 16 + 4; i++)

                            if (h->non_zero_count_cache[scan8[i]] ||

                                dctcoef_get(h->mb, PIXEL_SHIFT, i * 16))

                                idct_add(dest[j - 1] + block_offset[i],

                                         h->mb + (i * 16 << PIXEL_SHIFT),

                                         uvlinesize);

                        if (chroma422) {

                            for (i = j * 16 + 4; i < j * 16 + 8; i++)

                                if (h->non_zero_count_cache[scan8[i + 4]] ||

                                    dctcoef_get(h->mb, PIXEL_SHIFT, i * 16))

                                    idct_add(dest[j - 1] + block_offset[i + 4],

                                             h->mb + (i * 16 << PIXEL_SHIFT),

                                             uvlinesize);

                        }

                    }

                }

            } else {

                if (is_h264) {

                    int qp[2];

                    if (chroma422) {

                        qp[0] = h->chroma_qp[0] + 3;

                        qp[1] = h->chroma_qp[1] + 3;

                    } else {

                        qp[0] = h->chroma_qp[0];

                        qp[1] = h->chroma_qp[1];

                    }

                    if (h->non_zero_count_cache[scan8[CHROMA_DC_BLOCK_INDEX + 0]])

                        h->h264dsp.h264_chroma_dc_dequant_idct(h->mb + (16 * 16 * 1 << PIXEL_SHIFT),

                                                               h->dequant4_coeff[IS_INTRA(mb_type) ? 1 : 4][qp[0]][0]);

                    if (h->non_zero_count_cache[scan8[CHROMA_DC_BLOCK_INDEX + 1]])

                        h->h264dsp.h264_chroma_dc_dequant_idct(h->mb + (16 * 16 * 2 << PIXEL_SHIFT),

                                                               h->dequant4_coeff[IS_INTRA(mb_type) ? 2 : 5][qp[1]][0]);

                    h->h264dsp.h264_idct_add8(dest, block_offset,

                                              h->mb, uvlinesize,

                                              h->non_zero_count_cache);

                } else if (CONFIG_SVQ3_DECODER) {

                    h->h264dsp.h264_chroma_dc_dequant_idct(h->mb + 16 * 16 * 1,

                                                           h->dequant4_coeff[IS_INTRA(mb_type) ? 1 : 4][h->chroma_qp[0]][0]);

                    h->h264dsp.h264_chroma_dc_dequant_idct(h->mb + 16 * 16 * 2,

                                                           h->dequant4_coeff[IS_INTRA(mb_type) ? 2 : 5][h->chroma_qp[1]][0]);

                    for (j = 1; j < 3; j++) {

                        for (i = j * 16; i < j * 16 + 4; i++)

                            if (h->non_zero_count_cache[scan8[i]] || h->mb[i * 16]) {

                                uint8_t *const ptr = dest[j - 1] + block_offset[i];

                                ff_svq3_add_idct_c(ptr, h->mb + i * 16,

                                                   uvlinesize,

                                                   ff_h264_chroma_qp[0][h->qscale + 12] - 12, 2);

                            }

                    }

                }

            }

        }

        if (h->cbp || IS_INTRA(mb_type)) {

            h->dsp.clear_blocks(h->mb);

            h->dsp.clear_blocks(h->mb + (24 * 16 << PIXEL_SHIFT));

        }

    }

}
