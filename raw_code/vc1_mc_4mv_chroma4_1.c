static void vc1_mc_4mv_chroma4(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    DSPContext *dsp = &v->s.dsp;

    uint8_t *srcU, *srcV;

    int uvsrc_x, uvsrc_y;

    int uvmx_field[4], uvmy_field[4];

    int i, off, tx, ty;

    int fieldmv = v->blk_mv_type[s->block_index[0]];

    static const int s_rndtblfield[16] = { 0, 0, 1, 2, 4, 4, 5, 6, 2, 2, 3, 8, 6, 6, 7, 12 };

    int v_dist = fieldmv ? 1 : 4; // vertical offset for lower sub-blocks

    int v_edge_pos = s->v_edge_pos >> 1;



    if (!v->s.last_picture.f.data[0])

        return;

    if (s->flags & CODEC_FLAG_GRAY)

        return;



    for (i = 0; i < 4; i++) {

        tx = s->mv[0][i][0];

        uvmx_field[i] = (tx + ((tx & 3) == 3)) >> 1;

        ty = s->mv[0][i][1];

        if (fieldmv)

            uvmy_field[i] = (ty >> 4) * 8 + s_rndtblfield[ty & 0xF];

        else

            uvmy_field[i] = (ty + ((ty & 3) == 3)) >> 1;

    }



    for (i = 0; i < 4; i++) {

        off = (i & 1) * 4 + ((i & 2) ? v_dist * s->uvlinesize : 0);

        uvsrc_x = s->mb_x * 8 +  (i & 1) * 4           + (uvmx_field[i] >> 2);

        uvsrc_y = s->mb_y * 8 + ((i & 2) ? v_dist : 0) + (uvmy_field[i] >> 2);

        // FIXME: implement proper pull-back (see vc1cropmv.c, vc1CROPMV_ChromaPullBack())

        uvsrc_x = av_clip(uvsrc_x, -8, s->avctx->coded_width  >> 1);

        uvsrc_y = av_clip(uvsrc_y, -8, s->avctx->coded_height >> 1);

        srcU = s->last_picture.f.data[1] + uvsrc_y * s->uvlinesize + uvsrc_x;

        srcV = s->last_picture.f.data[2] + uvsrc_y * s->uvlinesize + uvsrc_x;

        uvmx_field[i] = (uvmx_field[i] & 3) << 1;

        uvmy_field[i] = (uvmy_field[i] & 3) << 1;



        if (fieldmv && !(uvsrc_y & 1))

            v_edge_pos--;

        if (fieldmv && (uvsrc_y & 1) && uvsrc_y < 2)

            uvsrc_y--;

        if ((v->mv_mode == MV_PMODE_INTENSITY_COMP)

            || s->h_edge_pos < 10 || v_edge_pos < (5 << fieldmv)

            || (unsigned)uvsrc_x > (s->h_edge_pos >> 1) - 5

            || (unsigned)uvsrc_y > v_edge_pos - (5 << fieldmv)) {

            s->dsp.emulated_edge_mc(s->edge_emu_buffer, srcU, s->uvlinesize,

                                    5, (5 << fieldmv), uvsrc_x, uvsrc_y,

                                    s->h_edge_pos >> 1, v_edge_pos);

            s->dsp.emulated_edge_mc(s->edge_emu_buffer + 16, srcV, s->uvlinesize,

                                    5, (5 << fieldmv), uvsrc_x, uvsrc_y,

                                    s->h_edge_pos >> 1, v_edge_pos);

            srcU = s->edge_emu_buffer;

            srcV = s->edge_emu_buffer + 16;



            /* if we deal with intensity compensation we need to scale source blocks */

            if (v->mv_mode == MV_PMODE_INTENSITY_COMP) {

                int i, j;

                uint8_t *src, *src2;



                src  = srcU;

                src2 = srcV;

                for (j = 0; j < 5; j++) {

                    for (i = 0; i < 5; i++) {

                        src[i]  = v->lutuv[src[i]];

                        src2[i] = v->lutuv[src2[i]];

                    }

                    src  += s->uvlinesize << 1;

                    src2 += s->uvlinesize << 1;

                }

            }

        }

        if (!v->rnd) {

            dsp->put_h264_chroma_pixels_tab[1](s->dest[1] + off, srcU, s->uvlinesize << fieldmv, 4, uvmx_field[i], uvmy_field[i]);

            dsp->put_h264_chroma_pixels_tab[1](s->dest[2] + off, srcV, s->uvlinesize << fieldmv, 4, uvmx_field[i], uvmy_field[i]);

        } else {

            v->vc1dsp.put_no_rnd_vc1_chroma_pixels_tab[1](s->dest[1] + off, srcU, s->uvlinesize << fieldmv, 4, uvmx_field[i], uvmy_field[i]);

            v->vc1dsp.put_no_rnd_vc1_chroma_pixels_tab[1](s->dest[2] + off, srcV, s->uvlinesize << fieldmv, 4, uvmx_field[i], uvmy_field[i]);

        }

    }

}
