static void interpolate_refplane(DiracContext *s, DiracFrame *ref, int plane, int width, int height)

{

    /* chroma allocates an edge of 8 when subsampled

       which for 4:2:2 means an h edge of 16 and v edge of 8

       just use 8 for everything for the moment */

    int i, edge = EDGE_WIDTH/2;



    ref->hpel[plane][0] = ref->avframe->data[plane];

    s->mpvencdsp.draw_edges(ref->hpel[plane][0], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM); /* EDGE_TOP | EDGE_BOTTOM values just copied to make it build, this needs to be ensured */



    /* no need for hpel if we only have fpel vectors */

    if (!s->mv_precision)

        return;



    for (i = 1; i < 4; i++) {

        if (!ref->hpel_base[plane][i])

            ref->hpel_base[plane][i] = av_malloc((height+2*edge) * ref->avframe->linesize[plane] + 32);

        /* we need to be 16-byte aligned even for chroma */

        ref->hpel[plane][i] = ref->hpel_base[plane][i] + edge*ref->avframe->linesize[plane] + 16;

    }



    if (!ref->interpolated[plane]) {

        s->diracdsp.dirac_hpel_filter(ref->hpel[plane][1], ref->hpel[plane][2],

                                      ref->hpel[plane][3], ref->hpel[plane][0],

                                      ref->avframe->linesize[plane], width, height);

        s->mpvencdsp.draw_edges(ref->hpel[plane][1], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(ref->hpel[plane][2], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);

        s->mpvencdsp.draw_edges(ref->hpel[plane][3], ref->avframe->linesize[plane], width, height, edge, edge, EDGE_TOP | EDGE_BOTTOM);

    }

    ref->interpolated[plane] = 1;

}
