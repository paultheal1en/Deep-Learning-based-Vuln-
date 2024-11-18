static void frame_end(MpegEncContext *s)

{

    int i;



    if (s->unrestricted_mv &&

        s->current_picture.reference &&

        !s->intra_only) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(s->avctx->pix_fmt);

        int hshift = desc->log2_chroma_w;

        int vshift = desc->log2_chroma_h;

        s->mpvencdsp.draw_edges(s->current_picture.f->data[0], s->linesize,

                                s->h_edge_pos, s->v_edge_pos,

                                EDGE_WIDTH, EDGE_WIDTH,

                                EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(s->current_picture.f->data[1], s->uvlinesize,

                                s->h_edge_pos >> hshift,

                                s->v_edge_pos >> vshift,

                                EDGE_WIDTH >> hshift,

                                EDGE_WIDTH >> vshift,

                                EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(s->current_picture.f->data[2], s->uvlinesize,

                                s->h_edge_pos >> hshift,

                                s->v_edge_pos >> vshift,

                                EDGE_WIDTH >> hshift,

                                EDGE_WIDTH >> vshift,

                                EDGE_TOP | EDGE_BOTTOM);

    }



    emms_c();



    s->last_pict_type                 = s->pict_type;

    s->last_lambda_for [s->pict_type] = s->current_picture_ptr->f->quality;

    if (s->pict_type!= AV_PICTURE_TYPE_B)

        s->last_non_b_pict_type = s->pict_type;



    if (s->encoding) {

        /* release non-reference frames */

        for (i = 0; i < MAX_PICTURE_COUNT; i++) {

            if (!s->picture[i].reference)

                ff_mpeg_unref_picture(s->avctx, &s->picture[i]);

        }

    }



    s->avctx->coded_frame = s->current_picture_ptr->f;



}
