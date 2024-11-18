int decode_splitmvs(VP8Context *s, VP56RangeCoder *c, VP8Macroblock *mb, int layout)

{

    int part_idx;

    int n, num;

    VP8Macroblock *top_mb;

    VP8Macroblock *left_mb = &mb[-1];

    const uint8_t *mbsplits_left = vp8_mbsplits[left_mb->partitioning];

    const uint8_t *mbsplits_top, *mbsplits_cur, *firstidx;

    VP56mv *top_mv;

    VP56mv *left_mv = left_mb->bmv;

    VP56mv *cur_mv  = mb->bmv;



    if (!layout) // layout is inlined, s->mb_layout is not

        top_mb = &mb[2];

    else

        top_mb = &mb[-s->mb_width - 1];

    mbsplits_top = vp8_mbsplits[top_mb->partitioning];

    top_mv       = top_mb->bmv;



    if (vp56_rac_get_prob_branchy(c, vp8_mbsplit_prob[0])) {

        if (vp56_rac_get_prob_branchy(c, vp8_mbsplit_prob[1]))

            part_idx = VP8_SPLITMVMODE_16x8 + vp56_rac_get_prob(c, vp8_mbsplit_prob[2]);

        else

            part_idx = VP8_SPLITMVMODE_8x8;

    } else {

        part_idx = VP8_SPLITMVMODE_4x4;

    }



    num              = vp8_mbsplit_count[part_idx];

    mbsplits_cur     = vp8_mbsplits[part_idx],

    firstidx         = vp8_mbfirstidx[part_idx];

    mb->partitioning = part_idx;



    for (n = 0; n < num; n++) {

        int k = firstidx[n];

        uint32_t left, above;

        const uint8_t *submv_prob;



        if (!(k & 3))

            left = AV_RN32A(&left_mv[mbsplits_left[k + 3]]);

        else

            left = AV_RN32A(&cur_mv[mbsplits_cur[k - 1]]);

        if (k <= 3)

            above = AV_RN32A(&top_mv[mbsplits_top[k + 12]]);

        else

            above = AV_RN32A(&cur_mv[mbsplits_cur[k - 4]]);



        submv_prob = get_submv_prob(left, above);



        if (vp56_rac_get_prob_branchy(c, submv_prob[0])) {

            if (vp56_rac_get_prob_branchy(c, submv_prob[1])) {

                if (vp56_rac_get_prob_branchy(c, submv_prob[2])) {

                    mb->bmv[n].y = mb->mv.y + read_mv_component(c, s->prob->mvc[0]);

                    mb->bmv[n].x = mb->mv.x + read_mv_component(c, s->prob->mvc[1]);

                } else {

                    AV_ZERO32(&mb->bmv[n]);

                }

            } else {

                AV_WN32A(&mb->bmv[n], above);

            }

        } else {

            AV_WN32A(&mb->bmv[n], left);

        }

    }



    return num;

}
