static int vp9_decode_frame(AVCodecContext *avctx, void *frame,

                            int *got_frame, AVPacket *pkt)

{

    const uint8_t *data = pkt->data;

    int size = pkt->size;

    VP9Context *s = avctx->priv_data;

    int ret, i, j, ref;

    int retain_segmap_ref = s->s.frames[REF_FRAME_SEGMAP].segmentation_map &&

                            (!s->s.h.segmentation.enabled || !s->s.h.segmentation.update_map);

    AVFrame *f;



    if ((ret = decode_frame_header(avctx, data, size, &ref)) < 0) {

        return ret;

    } else if (ret == 0) {

        if (!s->s.refs[ref].f->buf[0]) {

            av_log(avctx, AV_LOG_ERROR, "Requested reference %d not available\n", ref);

            return AVERROR_INVALIDDATA;

        }

        if ((ret = av_frame_ref(frame, s->s.refs[ref].f)) < 0)

            return ret;

        ((AVFrame *)frame)->pts = pkt->pts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

        ((AVFrame *)frame)->pkt_pts = pkt->pts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

        ((AVFrame *)frame)->pkt_dts = pkt->dts;

        for (i = 0; i < 8; i++) {

            if (s->next_refs[i].f->buf[0])

                ff_thread_release_buffer(avctx, &s->next_refs[i]);

            if (s->s.refs[i].f->buf[0] &&

                (ret = ff_thread_ref_frame(&s->next_refs[i], &s->s.refs[i])) < 0)

                return ret;

        }

        *got_frame = 1;

        return pkt->size;

    }

    data += ret;

    size -= ret;



    if (!retain_segmap_ref || s->s.h.keyframe || s->s.h.intraonly) {

        if (s->s.frames[REF_FRAME_SEGMAP].tf.f->buf[0])

            vp9_frame_unref(avctx, &s->s.frames[REF_FRAME_SEGMAP]);

        if (!s->s.h.keyframe && !s->s.h.intraonly && !s->s.h.errorres && s->s.frames[CUR_FRAME].tf.f->buf[0] &&

            (ret = vp9_frame_ref(avctx, &s->s.frames[REF_FRAME_SEGMAP], &s->s.frames[CUR_FRAME])) < 0)

            return ret;

    }

    if (s->s.frames[REF_FRAME_MVPAIR].tf.f->buf[0])

        vp9_frame_unref(avctx, &s->s.frames[REF_FRAME_MVPAIR]);

    if (!s->s.h.intraonly && !s->s.h.keyframe && !s->s.h.errorres && s->s.frames[CUR_FRAME].tf.f->buf[0] &&

        (ret = vp9_frame_ref(avctx, &s->s.frames[REF_FRAME_MVPAIR], &s->s.frames[CUR_FRAME])) < 0)

        return ret;

    if (s->s.frames[CUR_FRAME].tf.f->buf[0])

        vp9_frame_unref(avctx, &s->s.frames[CUR_FRAME]);

    if ((ret = vp9_frame_alloc(avctx, &s->s.frames[CUR_FRAME])) < 0)

        return ret;

    f = s->s.frames[CUR_FRAME].tf.f;

    f->key_frame = s->s.h.keyframe;

    f->pict_type = (s->s.h.keyframe || s->s.h.intraonly) ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;



    if (s->s.frames[REF_FRAME_SEGMAP].tf.f->buf[0] &&

        (s->s.frames[REF_FRAME_MVPAIR].tf.f->width  != s->s.frames[CUR_FRAME].tf.f->width ||

         s->s.frames[REF_FRAME_MVPAIR].tf.f->height != s->s.frames[CUR_FRAME].tf.f->height)) {

        vp9_frame_unref(avctx, &s->s.frames[REF_FRAME_SEGMAP]);

    }



    // ref frame setup

    for (i = 0; i < 8; i++) {

        if (s->next_refs[i].f->buf[0])

            ff_thread_release_buffer(avctx, &s->next_refs[i]);

        if (s->s.h.refreshrefmask & (1 << i)) {

            ret = ff_thread_ref_frame(&s->next_refs[i], &s->s.frames[CUR_FRAME].tf);

        } else if (s->s.refs[i].f->buf[0]) {

            ret = ff_thread_ref_frame(&s->next_refs[i], &s->s.refs[i]);

        }

        if (ret < 0)

            return ret;

    }



    if (avctx->hwaccel) {

        ret = avctx->hwaccel->start_frame(avctx, NULL, 0);

        if (ret < 0)

            return ret;

        ret = avctx->hwaccel->decode_slice(avctx, pkt->data, pkt->size);

        if (ret < 0)

            return ret;

        ret = avctx->hwaccel->end_frame(avctx);

        if (ret < 0)

            return ret;

        goto finish;

    }



    // main tile decode loop

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

        avctx->active_thread_type == FF_THREAD_FRAME && s->s.h.refreshctx && !s->s.h.parallelmode;

    if ((ret = update_block_buffers(avctx)) < 0) {

        av_log(avctx, AV_LOG_ERROR,

               "Failed to allocate block buffers\n");

        return ret;

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

        ff_thread_finish_setup(avctx);

    } else if (!s->s.h.refreshctx) {

        ff_thread_finish_setup(avctx);

    }



#if HAVE_THREADS

    if (avctx->active_thread_type & FF_THREAD_SLICE) {

        for (i = 0; i < s->sb_rows; i++)

            atomic_store(&s->entries[i], 0);

    }

#endif



    do {

        for (i = 0; i < s->active_tile_cols; i++) {

            s->td[i].b = s->td[i].b_base;

            s->td[i].block = s->td[i].block_base;

            s->td[i].uvblock[0] = s->td[i].uvblock_base[0];

            s->td[i].uvblock[1] = s->td[i].uvblock_base[1];

            s->td[i].eob = s->td[i].eob_base;

            s->td[i].uveob[0] = s->td[i].uveob_base[0];

            s->td[i].uveob[1] = s->td[i].uveob_base[1];

        }



#if HAVE_THREADS

        if (avctx->active_thread_type == FF_THREAD_SLICE) {

            int tile_row, tile_col;



            av_assert1(!s->pass);



            for (tile_row = 0; tile_row < s->s.h.tiling.tile_rows; tile_row++) {

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

                    if (tile_size > size)

                        return AVERROR_INVALIDDATA;

                    ret = ff_vp56_init_range_decoder(&s->td[tile_col].c_b[tile_row], data, tile_size);

                    if (ret < 0)

                        return ret;

                    if (vp56_rac_get_prob_branchy(&s->td[tile_col].c_b[tile_row], 128)) // marker bit

                        return AVERROR_INVALIDDATA;

                    data += tile_size;

                    size -= tile_size;

                }

            }



            ff_slice_thread_execute_with_mainfunc(avctx, decode_tiles_mt, loopfilter_proc, s->td, NULL, s->s.h.tiling.tile_cols);

        } else

#endif

        {

            ret = decode_tiles(avctx, data, size);

            if (ret < 0)

                return ret;

        }



        // Sum all counts fields into td[0].counts for tile threading

        if (avctx->active_thread_type == FF_THREAD_SLICE)

            for (i = 1; i < s->s.h.tiling.tile_cols; i++)

                for (j = 0; j < sizeof(s->td[i].counts) / sizeof(unsigned); j++)

                    ((unsigned *)&s->td[0].counts)[j] += ((unsigned *)&s->td[i].counts)[j];



        if (s->pass < 2 && s->s.h.refreshctx && !s->s.h.parallelmode) {

            ff_vp9_adapt_probs(s);

            ff_thread_finish_setup(avctx);

        }

    } while (s->pass++ == 1);

    ff_thread_report_progress(&s->s.frames[CUR_FRAME].tf, INT_MAX, 0);



finish:

    // ref frame setup

    for (i = 0; i < 8; i++) {

        if (s->s.refs[i].f->buf[0])

            ff_thread_release_buffer(avctx, &s->s.refs[i]);

        if (s->next_refs[i].f->buf[0] &&

            (ret = ff_thread_ref_frame(&s->s.refs[i], &s->next_refs[i])) < 0)

            return ret;

    }



    if (!s->s.h.invisible) {

        if ((ret = av_frame_ref(frame, s->s.frames[CUR_FRAME].tf.f)) < 0)

            return ret;

        *got_frame = 1;

    }



    return pkt->size;

}
