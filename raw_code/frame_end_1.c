static void frame_end(MpegEncContext *s)

{

    if (s->unrestricted_mv &&

        s->current_picture.reference &&

        !s->intra_only) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(s->avctx->pix_fmt);

        int hshift = desc->log2_chroma_w;

        int vshift = desc->log2_chroma_h;

        s->mpvencdsp.draw_edges(s->current_picture.f->data[0],

                                s->current_picture.f->linesize[0],

                                s->h_edge_pos, s->v_edge_pos,

                                EDGE_WIDTH, EDGE_WIDTH,

                                EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(s->current_picture.f->data[1],

                                s->current_picture.f->linesize[1],

                                s->h_edge_pos >> hshift,

                                s->v_edge_pos >> vshift,

                                EDGE_WIDTH >> hshift,

                                EDGE_WIDTH >> vshift,

                                EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(s->current_picture.f->data[2],

                                s->current_picture.f->linesize[2],

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



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS


    av_frame_copy_props(s->avctx->coded_frame, s->current_picture.f);

FF_ENABLE_DEPRECATION_WARNINGS

#endif

#if FF_API_ERROR_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    memcpy(s->current_picture.f->error, s->current_picture.encoding_error,

           sizeof(s->current_picture.encoding_error));

FF_ENABLE_DEPRECATION_WARNINGS

#endif

}