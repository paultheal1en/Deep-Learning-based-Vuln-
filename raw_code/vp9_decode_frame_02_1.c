static int vp9_decode_frame(AVCodecContext *ctx, void *frame,

                            int *got_frame, AVPacket *pkt)

{

    const uint8_t *data = pkt->data;

    int size = pkt->size;

    VP9Context *s = ctx->priv_data;

    int res, tile_row, tile_col, i, ref, row, col;

    int retain_segmap_ref = s->s.frames[REF_FRAME_SEGMAP].segmentation_map &&

                            (!s->s.h.segmentation.enabled || !s->s.h.segmentation.update_map);

    ptrdiff_t yoff, uvoff, ls_y, ls_uv;

    AVFrame *f;

    int bytesperpixel;



    if ((res = decode_frame_header(ctx, data, size, &ref)) < 0) {

        return res;

    } else if (res == 0) {

        if (!s->s.refs[ref].f->buf[0]) {

            av_log(ctx, AV_LOG_ERROR, "Requested reference %d not available\n", ref);

            return AVERROR_INVALIDDATA;

        }

        if ((res = av_frame_ref(frame, s->s.refs[ref].f)) < 0)

            return res;

        ((AVFrame *)frame)->pts = pkt->pts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

        ((AVFrame *)frame)->pkt_pts = pkt->pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

        ((AVFrame *)frame)->pkt_dts = pkt->dts;

        for (i = 0; i < 8; i++) {

            if (s->next_refs[i].f->buf[0])

                ff_thread_release_buffer(ctx, &s->next_refs[i]);

            if (s->s.refs[i].f->buf[0] &&

                (res = ff_thread_ref_frame(&s->next_refs[i], &s->s.refs[i])) < 0)

                return res;

        }

        *got_frame = 1;

        return pkt->size;

    }

    data += res;

    size -= res;



    if (!retain_segmap_ref || s->s.h.keyframe || s->s.h.intraonly) {

        if (s->s.frames[REF_FRAME_SEGMAP].tf.f->buf[0])

            vp9_unref_frame(ctx, &s->s.frames[REF_FRAME_SEGMAP]);

        if (!s->s.h.keyframe && !s->s.h.intraonly && !s->s.h.errorres && s->s.frames[CUR_FRAME].tf.f->buf[0] &&

            (res = vp9_ref_frame(ctx, &s->s.frames[REF_FRAME_SEGMAP], &s->s.frames[CUR_FRAME])) < 0)

            return res;

    }

    if (s->s.frames[REF_FRAME_MVPAIR].tf.f->buf[0])

        vp9_unref_frame(ctx, &s->s.frames[REF_FRAME_MVPAIR]);

    if (!s->s.h.intraonly && !s->s.h.keyframe && !s->s.h.errorres && s->s.frames[CUR_FRAME].tf.f->buf[0] &&

        (res = vp9_ref_frame(ctx, &s->s.frames[REF_FRAME_MVPAIR], &s->s.frames[CUR_FRAME])) < 0)

        return res;

    if (s->s.frames[CUR_FRAME].tf.f->buf[0])

        vp9_unref_frame(ctx, &s->s.frames[CUR_FRAME]);

    if ((res = vp9_alloc_frame(ctx, &s->s.frames[CUR_FRAME])) < 0)

        return res;

    f = s->s.frames[CUR_FRAME].tf.f;

    f->key_frame = s->s.h.keyframe;

    f->pict_type = (s->s.h.keyframe || s->s.h.intraonly) ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;

    ls_y = f->linesize[0];

    ls_uv =f->linesize[1];



    if (s->s.frames[REF_FRAME_SEGMAP].tf.f->buf[0] &&

        (s->s.frames[REF_FRAME_MVPAIR].tf.f->width  != s->s.frames[CUR_FRAME].tf.f->width ||

         s->s.frames[REF_FRAME_MVPAIR].tf.f->height != s->s.frames[CUR_FRAME].tf.f->height)) {

        vp9_unref_frame(ctx, &s->s.frames[REF_FRAME_SEGMAP]);

    }



    // ref frame setup

    for (i = 0; i < 8; i++) {

        if (s->next_refs[i].f->buf[0])

            ff_thread_release_buffer(ctx, &s->next_refs[i]);

        if (s->s.h.refreshrefmask & (1 << i)) {

            res = ff_thread_ref_frame(&s->next_refs[i], &s->s.frames[CUR_FRAME].tf);

        } else if (s->s.refs[i].f->buf[0]) {

            res = ff_thread_ref_frame(&s->next_refs[i], &s->s.refs[i]);

        }

        if (res < 0)

            return res;

    }



    if (ctx->hwaccel) {

        res = ctx->hwaccel->start_frame(ctx, NULL, 0);

        if (res < 0)

            return res;

        res = ctx->hwaccel->decode_slice(ctx, pkt->data, pkt->size);

        if (res < 0)

            return res;

        res = ctx->hwaccel->end_frame(ctx);

        if (res < 0)

            return res;

        goto finish;

    }



    // main tile decode loop

    bytesperpixel = s->bytesperpixel;

    memset(s->above_partition_ctx, 0, s->cols);

    memset(s->above_skip_ctx, 0, s->cols);

    if (s->s.h.keyframe || s->s.h.intraonly) {

        memset(s->above_mode_ctx, DC_PRED, s->cols * 2);

    } else {

        memset(s->above_mode_ctx, NEARESTMV, s->cols);

    }

    memset(s->above_y_nnz_ctx, 0, s->sb_cols * 16);

    memset(s->above_uv_nnz_ctx[0], 0, s->sb_cols * 16 >> s->ss_h);

    memset(s->above_uv_nnz_ctx[1], 0, s->sb_cols * 16 >> s->ss_h);

    memset(s->above_segpred_ctx, 0, s->cols);

    s->pass = s->s.frames[CUR_FRAME].uses_2pass =

        ctx->active_thread_type == FF_THREAD_FRAME && s->s.h.refreshctx && !s->s.h.parallelmode;

    if ((res = update_block_buffers(ctx)) < 0) {

        av_log(ctx, AV_LOG_ERROR,

               "Failed to allocate block buffers\n");

        return res;

    }

    if (s->s.h.refreshctx && s->s.h.parallelmode) {

        int j, k, l, m;



        for (i = 0; i < 4; i++) {

            for (j = 0; j < 2; j++)

                for (k = 0; k < 2; k++)

                    for (l = 0; l < 6; l++)

                        for (m = 0; m < 6; m++)

                            memcpy(s->prob_ctx[s->s.h.framectxid].coef[i][j][k][l][m],

                                   s->prob.coef[i][j][k][l][m], 3);

            if (s->s.h.txfmmode == i)

                break;

        }

        s->prob_ctx[s->s.h.framectxid].p = s->prob.p;

        ff_thread_finish_setup(ctx);

    } else if (!s->s.h.refreshctx) {

        ff_thread_finish_setup(ctx);

    }



    do {

        yoff = uvoff = 0;

        s->b = s->b_base;

        s->block = s->block_base;

        s->uvblock[0] = s->uvblock_base[0];

        s->uvblock[1] = s->uvblock_base[1];

        s->eob = s->eob_base;

        s->uveob[0] = s->uveob_base[0];

        s->uveob[1] = s->uveob_base[1];



        for (tile_row = 0; tile_row < s->s.h.tiling.tile_rows; tile_row++) {

            set_tile_offset(&s->tile_row_start, &s->tile_row_end,

                            tile_row, s->s.h.tiling.log2_tile_rows, s->sb_rows);

            if (s->pass != 2) {

                for (tile_col = 0; tile_col < s->s.h.tiling.tile_cols; tile_col++) {

                    int64_t tile_size;



                    if (tile_col == s->s.h.tiling.tile_cols - 1 &&

                        tile_row == s->s.h.tiling.tile_rows - 1) {

                        tile_size = size;

                    } else {

                        tile_size = AV_RB32(data);

                        data += 4;

                        size -= 4;

                    }

                    if (tile_size > size) {

                        ff_thread_report_progress(&s->s.frames[CUR_FRAME].tf, INT_MAX, 0);

                        return AVERROR_INVALIDDATA;

                    }

                    ff_vp56_init_range_decoder(&s->c_b[tile_col], data, tile_size);

                    if (vp56_rac_get_prob_branchy(&s->c_b[tile_col], 128)) { // marker bit

                        ff_thread_report_progress(&s->s.frames[CUR_FRAME].tf, INT_MAX, 0);

                        return AVERROR_INVALIDDATA;

                    }

                    data += tile_size;

                    size -= tile_size;

                }

            }



            for (row = s->tile_row_start; row < s->tile_row_end;

                 row += 8, yoff += ls_y * 64, uvoff += ls_uv * 64 >> s->ss_v) {

                struct VP9Filter *lflvl_ptr = s->lflvl;

                ptrdiff_t yoff2 = yoff, uvoff2 = uvoff;



                for (tile_col = 0; tile_col < s->s.h.tiling.tile_cols; tile_col++) {

                    set_tile_offset(&s->tile_col_start, &s->tile_col_end,

                                    tile_col, s->s.h.tiling.log2_tile_cols, s->sb_cols);



                    if (s->pass != 2) {

                        memset(s->left_partition_ctx, 0, 8);

                        memset(s->left_skip_ctx, 0, 8);

                        if (s->s.h.keyframe || s->s.h.intraonly) {

                            memset(s->left_mode_ctx, DC_PRED, 16);

                        } else {

                            memset(s->left_mode_ctx, NEARESTMV, 8);

                        }

                        memset(s->left_y_nnz_ctx, 0, 16);

                        memset(s->left_uv_nnz_ctx, 0, 32);

                        memset(s->left_segpred_ctx, 0, 8);



                        memcpy(&s->c, &s->c_b[tile_col], sizeof(s->c));

                    }



                    for (col = s->tile_col_start;

                         col < s->tile_col_end;

                         col += 8, yoff2 += 64 * bytesperpixel,

                         uvoff2 += 64 * bytesperpixel >> s->ss_h, lflvl_ptr++) {

                        // FIXME integrate with lf code (i.e. zero after each

                        // use, similar to invtxfm coefficients, or similar)

                        if (s->pass != 1) {

                            memset(lflvl_ptr->mask, 0, sizeof(lflvl_ptr->mask));

                        }



                        if (s->pass == 2) {

                            decode_sb_mem(ctx, row, col, lflvl_ptr,

                                          yoff2, uvoff2, BL_64X64);

                        } else {

                            decode_sb(ctx, row, col, lflvl_ptr,

                                      yoff2, uvoff2, BL_64X64);

                        }

                    }

                    if (s->pass != 2) {

                        memcpy(&s->c_b[tile_col], &s->c, sizeof(s->c));

                    }

                }



                if (s->pass == 1) {

                    continue;

                }



                // backup pre-loopfilter reconstruction data for intra

                // prediction of next row of sb64s

                if (row + 8 < s->rows) {

                    memcpy(s->intra_pred_data[0],

                           f->data[0] + yoff + 63 * ls_y,

                           8 * s->cols * bytesperpixel);

                    memcpy(s->intra_pred_data[1],

                           f->data[1] + uvoff + ((64 >> s->ss_v) - 1) * ls_uv,

                           8 * s->cols * bytesperpixel >> s->ss_h);

                    memcpy(s->intra_pred_data[2],

                           f->data[2] + uvoff + ((64 >> s->ss_v) - 1) * ls_uv,

                           8 * s->cols * bytesperpixel >> s->ss_h);

                }



                // loopfilter one row

                if (s->s.h.filter.level) {

                    yoff2 = yoff;

                    uvoff2 = uvoff;

                    lflvl_ptr = s->lflvl;

                    for (col = 0; col < s->cols;

                         col += 8, yoff2 += 64 * bytesperpixel,

                         uvoff2 += 64 * bytesperpixel >> s->ss_h, lflvl_ptr++) {

                        loopfilter_sb(ctx, lflvl_ptr, row, col, yoff2, uvoff2);

                    }

                }



                // FIXME maybe we can make this more finegrained by running the

                // loopfilter per-block instead of after each sbrow

                // In fact that would also make intra pred left preparation easier?

                ff_thread_report_progress(&s->s.frames[CUR_FRAME].tf, row >> 3, 0);

            }

        }



        if (s->pass < 2 && s->s.h.refreshctx && !s->s.h.parallelmode) {

            adapt_probs(s);

            ff_thread_finish_setup(ctx);

        }

    } while (s->pass++ == 1);

    ff_thread_report_progress(&s->s.frames[CUR_FRAME].tf, INT_MAX, 0);



finish:

    // ref frame setup

    for (i = 0; i < 8; i++) {

        if (s->s.refs[i].f->buf[0])

            ff_thread_release_buffer(ctx, &s->s.refs[i]);

        if (s->next_refs[i].f->buf[0] &&

            (res = ff_thread_ref_frame(&s->s.refs[i], &s->next_refs[i])) < 0)

            return res;

    }



    if (!s->s.h.invisible) {

        if ((res = av_frame_ref(frame, s->s.frames[CUR_FRAME].tf.f)) < 0)

            return res;

        *got_frame = 1;

    }



    return pkt->size;

}
