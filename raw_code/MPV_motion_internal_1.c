static av_always_inline void MPV_motion_internal(MpegEncContext *s,

                                                 uint8_t *dest_y,

                                                 uint8_t *dest_cb,

                                                 uint8_t *dest_cr,

                                                 int dir,

                                                 uint8_t **ref_picture,

                                                 op_pixels_func (*pix_op)[4],

                                                 qpel_mc_func (*qpix_op)[16],

                                                 int is_mpeg12)

{

    int i;

    int mb_y = s->mb_y;



    prefetch_motion(s, ref_picture, dir);



    if (!is_mpeg12 && s->obmc && s->pict_type != AV_PICTURE_TYPE_B) {

        apply_obmc(s, dest_y, dest_cb, dest_cr, ref_picture, pix_op);

        return;

    }



    switch (s->mv_type) {

    case MV_TYPE_16X16:

        if (s->mcsel) {

            if (s->real_sprite_warping_points == 1) {

                gmc1_motion(s, dest_y, dest_cb, dest_cr,

                            ref_picture);

            } else {

                gmc_motion(s, dest_y, dest_cb, dest_cr,

                           ref_picture);

            }

        } else if (!is_mpeg12 && s->quarter_sample) {

            qpel_motion(s, dest_y, dest_cb, dest_cr,

                        0, 0, 0,

                        ref_picture, pix_op, qpix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16);

        } else if (!is_mpeg12 && (CONFIG_WMV2_DECODER || CONFIG_WMV2_ENCODER) &&

                   s->mspel && s->codec_id == AV_CODEC_ID_WMV2) {

            ff_mspel_motion(s, dest_y, dest_cb, dest_cr,

                            ref_picture, pix_op,

                            s->mv[dir][0][0], s->mv[dir][0][1], 16);

        } else {

            mpeg_motion(s, dest_y, dest_cb, dest_cr, 0,

                        ref_picture, pix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16, mb_y);

        }

        break;

    case MV_TYPE_8X8:

        if (!is_mpeg12)

            apply_8x8(s, dest_y, dest_cb, dest_cr,

                      dir, ref_picture, qpix_op, pix_op);

        break;

    case MV_TYPE_FIELD:

        if (s->picture_structure == PICT_FRAME) {

            if (!is_mpeg12 && s->quarter_sample) {

                for (i = 0; i < 2; i++)

                    qpel_motion(s, dest_y, dest_cb, dest_cr,

                                1, i, s->field_select[dir][i],

                                ref_picture, pix_op, qpix_op,

                                s->mv[dir][i][0], s->mv[dir][i][1], 8);

            } else {

                /* top field */

                mpeg_motion_field(s, dest_y, dest_cb, dest_cr,

                                  0, s->field_select[dir][0],

                                  ref_picture, pix_op,

                                  s->mv[dir][0][0], s->mv[dir][0][1], 8, mb_y);

                /* bottom field */

                mpeg_motion_field(s, dest_y, dest_cb, dest_cr,

                                  1, s->field_select[dir][1],

                                  ref_picture, pix_op,

                                  s->mv[dir][1][0], s->mv[dir][1][1], 8, mb_y);

            }

        } else {

            if (s->picture_structure != s->field_select[dir][0] + 1 &&

                s->pict_type != AV_PICTURE_TYPE_B && !s->first_field) {

                ref_picture = s->current_picture_ptr->f.data;

            }



            mpeg_motion(s, dest_y, dest_cb, dest_cr,

                        s->field_select[dir][0],

                        ref_picture, pix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16, mb_y >> 1);

        }

        break;

    case MV_TYPE_16X8:

        for (i = 0; i < 2; i++) {

            uint8_t **ref2picture;



            if (s->picture_structure == s->field_select[dir][i] + 1

                || s->pict_type == AV_PICTURE_TYPE_B || s->first_field) {

                ref2picture = ref_picture;

            } else {

                ref2picture = s->current_picture_ptr->f.data;

            }



            mpeg_motion(s, dest_y, dest_cb, dest_cr,

                        s->field_select[dir][i],

                        ref2picture, pix_op,

                        s->mv[dir][i][0], s->mv[dir][i][1] + 16 * i,

                        8, mb_y >> 1);



            dest_y  += 16 * s->linesize;

            dest_cb += (16 >> s->chroma_y_shift) * s->uvlinesize;

            dest_cr += (16 >> s->chroma_y_shift) * s->uvlinesize;

        }

        break;

    case MV_TYPE_DMV:

        if (s->picture_structure == PICT_FRAME) {

            for (i = 0; i < 2; i++) {

                int j;

                for (j = 0; j < 2; j++)

                    mpeg_motion_field(s, dest_y, dest_cb, dest_cr,

                                      j, j ^ i, ref_picture, pix_op,

                                      s->mv[dir][2 * i + j][0],

                                      s->mv[dir][2 * i + j][1], 8, mb_y);

                pix_op = s->hdsp.avg_pixels_tab;

            }

        } else {

            for (i = 0; i < 2; i++) {

                mpeg_motion(s, dest_y, dest_cb, dest_cr,

                            s->picture_structure != i + 1,

                            ref_picture, pix_op,

                            s->mv[dir][2 * i][0], s->mv[dir][2 * i][1],

                            16, mb_y >> 1);



                // after put we make avg of the same block

                pix_op = s->hdsp.avg_pixels_tab;



                /* opposite parity is always in the same frame if this is

                 * second field */

                if (!s->first_field) {

                    ref_picture = s->current_picture_ptr->f.data;

                }

            }

        }

        break;

    default: assert(0);

    }

}
