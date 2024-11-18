void ff_MPV_frame_end(MpegEncContext *s)

{

    int i;

    /* redraw edges for the frame if decoding didn't complete */

    // just to make sure that all data is rendered.

    if (CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration) {

        ff_xvmc_field_end(s);

   } else if((s->error_count || s->encoding || !(s->avctx->codec->capabilities&CODEC_CAP_DRAW_HORIZ_BAND)) &&

              !s->avctx->hwaccel &&

              !(s->avctx->codec->capabilities & CODEC_CAP_HWACCEL_VDPAU) &&

              s->unrestricted_mv &&

              s->current_picture.f.reference &&

              !s->intra_only &&

              !(s->flags & CODEC_FLAG_EMU_EDGE)) {

        int hshift = av_pix_fmt_descriptors[s->avctx->pix_fmt].log2_chroma_w;

        int vshift = av_pix_fmt_descriptors[s->avctx->pix_fmt].log2_chroma_h;

        s->dsp.draw_edges(s->current_picture.f.data[0], s->current_picture.f.linesize[0],

                          s->h_edge_pos, s->v_edge_pos,

                          EDGE_WIDTH, EDGE_WIDTH,

                          EDGE_TOP | EDGE_BOTTOM);

        s->dsp.draw_edges(s->current_picture.f.data[1], s->current_picture.f.linesize[1],

                          s->h_edge_pos >> hshift, s->v_edge_pos >> vshift,

                          EDGE_WIDTH >> hshift, EDGE_WIDTH >> vshift,

                          EDGE_TOP | EDGE_BOTTOM);

        s->dsp.draw_edges(s->current_picture.f.data[2], s->current_picture.f.linesize[2],

                          s->h_edge_pos >> hshift, s->v_edge_pos >> vshift,

                          EDGE_WIDTH >> hshift, EDGE_WIDTH >> vshift,

                          EDGE_TOP | EDGE_BOTTOM);

    }



    emms_c();



    s->last_pict_type                 = s->pict_type;

    s->last_lambda_for [s->pict_type] = s->current_picture_ptr->f.quality;

    if (s->pict_type!= AV_PICTURE_TYPE_B) {

        s->last_non_b_pict_type = s->pict_type;

    }

#if 0

    /* copy back current_picture variables */

    for (i = 0; i < MAX_PICTURE_COUNT; i++) {

        if (s->picture[i].f.data[0] == s->current_picture.f.data[0]) {

            s->picture[i] = s->current_picture;

            break;

        }

    }

    assert(i < MAX_PICTURE_COUNT);

#endif



    if (s->encoding) {

        /* release non-reference frames */

        for (i = 0; i < s->picture_count; i++) {

            if (s->picture[i].f.data[0] && !s->picture[i].f.reference

                /* && s->picture[i].type != FF_BUFFER_TYPE_SHARED */) {

                free_frame_buffer(s, &s->picture[i]);

            }

        }

    }

    // clear copies, to avoid confusion

#if 0

    memset(&s->last_picture,    0, sizeof(Picture));

    memset(&s->next_picture,    0, sizeof(Picture));

    memset(&s->current_picture, 0, sizeof(Picture));

#endif

    s->avctx->coded_frame = &s->current_picture_ptr->f;



    if (s->codec_id != AV_CODEC_ID_H264 && s->current_picture.f.reference) {

        ff_thread_report_progress(&s->current_picture_ptr->f, INT_MAX, 0);

    }

}
