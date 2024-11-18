static av_always_inline void vc1_apply_p_v_loop_filter(VC1Context *v, int block_num)

{

    MpegEncContext *s  = &v->s;

    int mb_cbp         = v->cbp[s->mb_x - s->mb_stride],

        block_cbp      = mb_cbp      >> (block_num * 4), bottom_cbp,

        mb_is_intra    = v->is_intra[s->mb_x - s->mb_stride],

        block_is_intra = mb_is_intra >> (block_num * 4), bottom_is_intra;

    int idx, linesize  = block_num > 3 ? s->uvlinesize : s->linesize, ttblk;

    uint8_t *dst;



    if (block_num > 3) {

        dst      = s->dest[block_num - 3];

    } else {

        dst      = s->dest[0] + (block_num & 1) * 8 + ((block_num & 2) * 4 - 8) * linesize;

    }

    if (s->mb_y != s->end_mb_y || block_num < 2) {

        int16_t (*mv)[2];

        int mv_stride;



        if (block_num > 3) {

            bottom_cbp      = v->cbp[s->mb_x]      >> (block_num * 4);

            bottom_is_intra = v->is_intra[s->mb_x] >> (block_num * 4);

            mv              = &v->luma_mv[s->mb_x - s->mb_stride];

            mv_stride       = s->mb_stride;

        } else {

            bottom_cbp      = (block_num < 2) ? (mb_cbp               >> ((block_num + 2) * 4))

                                              : (v->cbp[s->mb_x]      >> ((block_num - 2) * 4));

            bottom_is_intra = (block_num < 2) ? (mb_is_intra          >> ((block_num + 2) * 4))

                                              : (v->is_intra[s->mb_x] >> ((block_num - 2) * 4));

            mv_stride       = s->b8_stride;

            mv              = &s->current_picture.motion_val[0][s->block_index[block_num] - 2 * mv_stride];

        }



        if (bottom_is_intra & 1 || block_is_intra & 1 ||

            mv[0][0] != mv[mv_stride][0] || mv[0][1] != mv[mv_stride][1]) {

            v->vc1dsp.vc1_v_loop_filter8(dst, linesize, v->pq);

        } else {

            idx = ((bottom_cbp >> 2) | block_cbp) & 3;

            if (idx == 3) {

                v->vc1dsp.vc1_v_loop_filter8(dst, linesize, v->pq);

            } else if (idx) {

                if (idx == 1)

                    v->vc1dsp.vc1_v_loop_filter4(dst + 4, linesize, v->pq);

                else

                    v->vc1dsp.vc1_v_loop_filter4(dst,     linesize, v->pq);

            }

        }

    }



    dst -= 4 * linesize;

    ttblk = (v->ttblk[s->mb_x - s->mb_stride] >> (block_num * 4)) & 0xF;

    if (ttblk == TT_4X4 || ttblk == TT_8X4) {

        idx = (block_cbp | (block_cbp >> 2)) & 3;

        if (idx == 3) {

            v->vc1dsp.vc1_v_loop_filter8(dst, linesize, v->pq);

        } else if (idx) {

            if (idx == 1)

                v->vc1dsp.vc1_v_loop_filter4(dst + 4, linesize, v->pq);

            else

                v->vc1dsp.vc1_v_loop_filter4(dst,     linesize, v->pq);

        }

    }

}
