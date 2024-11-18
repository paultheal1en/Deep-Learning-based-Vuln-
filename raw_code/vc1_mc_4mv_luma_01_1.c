static void vc1_mc_4mv_luma(VC1Context *v, int n, int dir, int avg)

{

    MpegEncContext *s = &v->s;

    uint8_t *srcY;

    int dxy, mx, my, src_x, src_y;

    int off;

    int fieldmv = (v->fcm == ILACE_FRAME) ? v->blk_mv_type[s->block_index[n]] : 0;

    int v_edge_pos = s->v_edge_pos >> v->field_mode;

    uint8_t (*luty)[256];

    int use_ic;



    if ((!v->field_mode ||

         (v->ref_field_type[dir] == 1 && v->cur_field_type == 1)) &&

        !v->s.last_picture.f.data[0])

        return;



    mx = s->mv[dir][n][0];

    my = s->mv[dir][n][1];



    if (!dir) {

        if (v->field_mode && (v->cur_field_type != v->ref_field_type[dir]) && v->second_field) {

            srcY = s->current_picture.f.data[0];

            luty = v->curr_luty;

            use_ic = v->curr_use_ic;

        } else {

            srcY = s->last_picture.f.data[0];

            luty = v->last_luty;

            use_ic = v->last_use_ic;

        }

    } else {

        srcY = s->next_picture.f.data[0];

        luty = v->next_luty;

        use_ic = v->next_use_ic;

    }



    if (!srcY) {

        av_log(v->s.avctx, AV_LOG_ERROR, "Referenced frame missing.\n");

        return;

    }



    if (v->field_mode) {

        if (v->cur_field_type != v->ref_field_type[dir])

            my = my - 2 + 4 * v->cur_field_type;

    }



    if (s->pict_type == AV_PICTURE_TYPE_P && n == 3 && v->field_mode) {

        int same_count = 0, opp_count = 0, k;

        int chosen_mv[2][4][2], f;

        int tx, ty;

        for (k = 0; k < 4; k++) {

            f = v->mv_f[0][s->block_index[k] + v->blocks_off];

            chosen_mv[f][f ? opp_count : same_count][0] = s->mv[0][k][0];

            chosen_mv[f][f ? opp_count : same_count][1] = s->mv[0][k][1];

            opp_count  += f;

            same_count += 1 - f;

        }

        f = opp_count > same_count;

        switch (f ? opp_count : same_count) {

        case 4:

            tx = median4(chosen_mv[f][0][0], chosen_mv[f][1][0],

                         chosen_mv[f][2][0], chosen_mv[f][3][0]);

            ty = median4(chosen_mv[f][0][1], chosen_mv[f][1][1],

                         chosen_mv[f][2][1], chosen_mv[f][3][1]);

            break;

        case 3:

            tx = mid_pred(chosen_mv[f][0][0], chosen_mv[f][1][0], chosen_mv[f][2][0]);

            ty = mid_pred(chosen_mv[f][0][1], chosen_mv[f][1][1], chosen_mv[f][2][1]);

            break;

        case 2:

            tx = (chosen_mv[f][0][0] + chosen_mv[f][1][0]) / 2;

            ty = (chosen_mv[f][0][1] + chosen_mv[f][1][1]) / 2;

            break;

        }

        s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][0] = tx;

        s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][1] = ty;

        for (k = 0; k < 4; k++)

            v->mv_f[1][s->block_index[k] + v->blocks_off] = f;

    }



    if (v->fcm == ILACE_FRAME) {  // not sure if needed for other types of picture

        int qx, qy;

        int width  = s->avctx->coded_width;

        int height = s->avctx->coded_height >> 1;

        if (s->pict_type == AV_PICTURE_TYPE_P) {

            s->current_picture.motion_val[1][s->block_index[n] + v->blocks_off][0] = mx;

            s->current_picture.motion_val[1][s->block_index[n] + v->blocks_off][1] = my;

        }

        qx = (s->mb_x * 16) + (mx >> 2);

        qy = (s->mb_y *  8) + (my >> 3);



        if (qx < -17)

            mx -= 4 * (qx + 17);

        else if (qx > width)

            mx -= 4 * (qx - width);

        if (qy < -18)

            my -= 8 * (qy + 18);

        else if (qy > height + 1)

            my -= 8 * (qy - height - 1);

    }



    if ((v->fcm == ILACE_FRAME) && fieldmv)

        off = ((n > 1) ? s->linesize : 0) + (n & 1) * 8;

    else

        off = s->linesize * 4 * (n & 2) + (n & 1) * 8;



    src_x = s->mb_x * 16 + (n & 1) * 8 + (mx >> 2);

    if (!fieldmv)

        src_y = s->mb_y * 16 + (n & 2) * 4 + (my >> 2);

    else

        src_y = s->mb_y * 16 + ((n > 1) ? 1 : 0) + (my >> 2);



    if (v->profile != PROFILE_ADVANCED) {

        src_x = av_clip(src_x, -16, s->mb_width  * 16);

        src_y = av_clip(src_y, -16, s->mb_height * 16);

    } else {

        src_x = av_clip(src_x, -17, s->avctx->coded_width);

        if (v->fcm == ILACE_FRAME) {

            if (src_y & 1)

                src_y = av_clip(src_y, -17, s->avctx->coded_height + 1);

            else

                src_y = av_clip(src_y, -18, s->avctx->coded_height);

        } else {

            src_y = av_clip(src_y, -18, s->avctx->coded_height + 1);

        }

    }



    srcY += src_y * s->linesize + src_x;

    if (v->field_mode && v->ref_field_type[dir])

        srcY += s->current_picture_ptr->f.linesize[0];



    if (fieldmv && !(src_y & 1))

        v_edge_pos--;

    if (fieldmv && (src_y & 1) && src_y < 4)

        src_y--;

    if (v->rangeredfrm || use_ic

        || s->h_edge_pos < 13 || v_edge_pos < 23

        || (unsigned)(src_x - s->mspel) > s->h_edge_pos - (mx & 3) - 8 - s->mspel * 2

        || (unsigned)(src_y - (s->mspel << fieldmv)) > v_edge_pos - (my & 3) - ((8 + s->mspel * 2) << fieldmv)) {

        srcY -= s->mspel * (1 + (s->linesize << fieldmv));

        /* check emulate edge stride and offset */

        s->vdsp.emulated_edge_mc(s->edge_emu_buffer, srcY,

                                 s->linesize, s->linesize,

                                 9 + s->mspel * 2, (9 + s->mspel * 2) << fieldmv,

                                 src_x - s->mspel, src_y - (s->mspel << fieldmv),

                                 s->h_edge_pos, v_edge_pos);

        srcY = s->edge_emu_buffer;

        /* if we deal with range reduction we need to scale source blocks */

        if (v->rangeredfrm) {

            int i, j;

            uint8_t *src;



            src = srcY;

            for (j = 0; j < 9 + s->mspel * 2; j++) {

                for (i = 0; i < 9 + s->mspel * 2; i++)

                    src[i] = ((src[i] - 128) >> 1) + 128;

                src += s->linesize << fieldmv;

            }

        }

        /* if we deal with intensity compensation we need to scale source blocks */

        if (use_ic) {

            int i, j;

            uint8_t *src;



            src = srcY;

            for (j = 0; j < 9 + s->mspel * 2; j++) {

                int f = v->field_mode ? v->ref_field_type[dir] : (((j<<fieldmv)+src_y - (s->mspel << fieldmv)) & 1);

                for (i = 0; i < 9 + s->mspel * 2; i++)

                    src[i] = luty[f][src[i]];

                src += s->linesize << fieldmv;

            }

        }

        srcY += s->mspel * (1 + (s->linesize << fieldmv));

    }



    if (s->mspel) {

        dxy = ((my & 3) << 2) | (mx & 3);

        if (avg)

            v->vc1dsp.avg_vc1_mspel_pixels_tab[dxy](s->dest[0] + off, srcY, s->linesize << fieldmv, v->rnd);

        else

            v->vc1dsp.put_vc1_mspel_pixels_tab[dxy](s->dest[0] + off, srcY, s->linesize << fieldmv, v->rnd);

    } else { // hpel mc - always used for luma

        dxy = (my & 2) | ((mx & 2) >> 1);

        if (!v->rnd)

            s->hdsp.put_pixels_tab[1][dxy](s->dest[0] + off, srcY, s->linesize, 8);

        else

            s->hdsp.put_no_rnd_pixels_tab[1][dxy](s->dest[0] + off, srcY, s->linesize, 8);

    }

}
