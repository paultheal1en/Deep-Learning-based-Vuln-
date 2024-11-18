static av_always_inline void MPV_motion_internal(MpegEncContext *s,

                              uint8_t *dest_y, uint8_t *dest_cb,

                              uint8_t *dest_cr, int dir,

                              uint8_t **ref_picture,

                              op_pixels_func (*pix_op)[4],

                              qpel_mc_func (*qpix_op)[16], int is_mpeg12)

{

    int dxy, mx, my, src_x, src_y, motion_x, motion_y;

    int mb_x, mb_y, i;

    uint8_t *ptr, *dest;



    mb_x = s->mb_x;

    mb_y = s->mb_y;



    prefetch_motion(s, ref_picture, dir);



    if(!is_mpeg12 && s->obmc && s->pict_type != AV_PICTURE_TYPE_B){

        LOCAL_ALIGNED_8(int16_t, mv_cache, [4], [4][2]);

        Picture *cur_frame = &s->current_picture;

        const int xy= s->mb_x + s->mb_y*s->mb_stride;

        const int mot_stride= s->b8_stride;

        const int mot_xy= mb_x*2 + mb_y*2*mot_stride;



        av_assert2(!s->mb_skipped);



        AV_COPY32(mv_cache[1][1], cur_frame->motion_val[0][mot_xy    ]);

        AV_COPY32(mv_cache[1][2], cur_frame->motion_val[0][mot_xy + 1]);



        AV_COPY32(mv_cache[2][1], cur_frame->motion_val[0][mot_xy + mot_stride    ]);

        AV_COPY32(mv_cache[2][2], cur_frame->motion_val[0][mot_xy + mot_stride + 1]);



        AV_COPY32(mv_cache[3][1], cur_frame->motion_val[0][mot_xy + mot_stride    ]);

        AV_COPY32(mv_cache[3][2], cur_frame->motion_val[0][mot_xy + mot_stride + 1]);



        if (mb_y == 0 || IS_INTRA(cur_frame->mb_type[xy - s->mb_stride])) {

            AV_COPY32(mv_cache[0][1], mv_cache[1][1]);

            AV_COPY32(mv_cache[0][2], mv_cache[1][2]);

        }else{

            AV_COPY32(mv_cache[0][1], cur_frame->motion_val[0][mot_xy - mot_stride    ]);

            AV_COPY32(mv_cache[0][2], cur_frame->motion_val[0][mot_xy - mot_stride + 1]);

        }



        if (mb_x == 0 || IS_INTRA(cur_frame->mb_type[xy - 1])) {

            AV_COPY32(mv_cache[1][0], mv_cache[1][1]);

            AV_COPY32(mv_cache[2][0], mv_cache[2][1]);

        }else{

            AV_COPY32(mv_cache[1][0], cur_frame->motion_val[0][mot_xy - 1]);

            AV_COPY32(mv_cache[2][0], cur_frame->motion_val[0][mot_xy - 1 + mot_stride]);

        }



        if (mb_x + 1 >= s->mb_width || IS_INTRA(cur_frame->mb_type[xy + 1])) {

            AV_COPY32(mv_cache[1][3], mv_cache[1][2]);

            AV_COPY32(mv_cache[2][3], mv_cache[2][2]);

        }else{

            AV_COPY32(mv_cache[1][3], cur_frame->motion_val[0][mot_xy + 2]);

            AV_COPY32(mv_cache[2][3], cur_frame->motion_val[0][mot_xy + 2 + mot_stride]);

        }



        mx = 0;

        my = 0;

        for(i=0;i<4;i++) {

            const int x= (i&1)+1;

            const int y= (i>>1)+1;

            int16_t mv[5][2]= {

                {mv_cache[y][x  ][0], mv_cache[y][x  ][1]},

                {mv_cache[y-1][x][0], mv_cache[y-1][x][1]},

                {mv_cache[y][x-1][0], mv_cache[y][x-1][1]},

                {mv_cache[y][x+1][0], mv_cache[y][x+1][1]},

                {mv_cache[y+1][x][0], mv_cache[y+1][x][1]}};

            //FIXME cleanup

            obmc_motion(s, dest_y + ((i & 1) * 8) + (i >> 1) * 8 * s->linesize,

                        ref_picture[0],

                        mb_x * 16 + (i & 1) * 8, mb_y * 16 + (i >>1) * 8,

                        pix_op[1],

                        mv);



            mx += mv[0][0];

            my += mv[0][1];

        }

        if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY))

            chroma_4mv_motion(s, dest_cb, dest_cr, ref_picture, pix_op[1], mx, my);



        return;

    }



    switch(s->mv_type) {

    case MV_TYPE_16X16:

        if(s->mcsel){

            if(s->real_sprite_warping_points==1){

                gmc1_motion(s, dest_y, dest_cb, dest_cr,

                            ref_picture);

            }else{

                gmc_motion(s, dest_y, dest_cb, dest_cr,

                            ref_picture);

            }

        }else if(!is_mpeg12 && s->quarter_sample){

            qpel_motion(s, dest_y, dest_cb, dest_cr,

                        0, 0, 0,

                        ref_picture, pix_op, qpix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16);

        } else if (!is_mpeg12 && (CONFIG_WMV2_DECODER || CONFIG_WMV2_ENCODER) &&

                   s->mspel && s->codec_id == AV_CODEC_ID_WMV2) {

            ff_mspel_motion(s, dest_y, dest_cb, dest_cr,

                        ref_picture, pix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16);

        }else

        {

            mpeg_motion(s, dest_y, dest_cb, dest_cr, 0,

                        ref_picture, pix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16, mb_y);

        }

        break;

    case MV_TYPE_8X8:

    if (!is_mpeg12) {

        mx = 0;

        my = 0;

        if(s->quarter_sample){

            for(i=0;i<4;i++) {

                motion_x = s->mv[dir][i][0];

                motion_y = s->mv[dir][i][1];



                dxy = ((motion_y & 3) << 2) | (motion_x & 3);

                src_x = mb_x * 16 + (motion_x >> 2) + (i & 1) * 8;

                src_y = mb_y * 16 + (motion_y >> 2) + (i >>1) * 8;



                /* WARNING: do no forget half pels */

                src_x = av_clip(src_x, -16, s->width);

                if (src_x == s->width)

                    dxy &= ~3;

                src_y = av_clip(src_y, -16, s->height);

                if (src_y == s->height)

                    dxy &= ~12;



                ptr = ref_picture[0] + (src_y * s->linesize) + (src_x);

                if(s->flags&CODEC_FLAG_EMU_EDGE){

                    if(   (unsigned)src_x > FFMAX(s->h_edge_pos - (motion_x&3) - 8, 0)

                       || (unsigned)src_y > FFMAX(s->v_edge_pos - (motion_y&3) - 8, 0)){

                        s->vdsp.emulated_edge_mc(s->edge_emu_buffer, ptr,

                                            s->linesize, 9, 9,

                                            src_x, src_y,

                                            s->h_edge_pos, s->v_edge_pos);

                        ptr= s->edge_emu_buffer;

                    }

                }

                dest = dest_y + ((i & 1) * 8) + (i >> 1) * 8 * s->linesize;

                qpix_op[1][dxy](dest, ptr, s->linesize);



                mx += s->mv[dir][i][0]/2;

                my += s->mv[dir][i][1]/2;

            }

        }else{

            for(i=0;i<4;i++) {

                hpel_motion(s, dest_y + ((i & 1) * 8) + (i >> 1) * 8 * s->linesize,

                            ref_picture[0],

                            mb_x * 16 + (i & 1) * 8, mb_y * 16 + (i >>1) * 8,

                            pix_op[1],

                            s->mv[dir][i][0], s->mv[dir][i][1]);



                mx += s->mv[dir][i][0];

                my += s->mv[dir][i][1];

            }

        }



        if(!CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY))

            chroma_4mv_motion(s, dest_cb, dest_cr, ref_picture, pix_op[1], mx, my);

    }

        break;

    case MV_TYPE_FIELD:

        if (s->picture_structure == PICT_FRAME) {

            if(!is_mpeg12 && s->quarter_sample){

                for(i=0; i<2; i++){

                    qpel_motion(s, dest_y, dest_cb, dest_cr,

                                1, i, s->field_select[dir][i],

                                ref_picture, pix_op, qpix_op,

                                s->mv[dir][i][0], s->mv[dir][i][1], 8);

                }

            }else{

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

            if(s->picture_structure != s->field_select[dir][0] + 1 && s->pict_type != AV_PICTURE_TYPE_B && !s->first_field){

                ref_picture = s->current_picture_ptr->f.data;

            }



            mpeg_motion(s, dest_y, dest_cb, dest_cr,

                        s->field_select[dir][0],

                        ref_picture, pix_op,

                        s->mv[dir][0][0], s->mv[dir][0][1], 16, mb_y>>1);

        }

        break;

    case MV_TYPE_16X8:

        for(i=0; i<2; i++){

            uint8_t ** ref2picture;



            if(s->picture_structure == s->field_select[dir][i] + 1

               || s->pict_type == AV_PICTURE_TYPE_B || s->first_field){

                ref2picture= ref_picture;

            }else{

                ref2picture = s->current_picture_ptr->f.data;

            }



            mpeg_motion(s, dest_y, dest_cb, dest_cr,

                        s->field_select[dir][i],

                        ref2picture, pix_op,

                        s->mv[dir][i][0], s->mv[dir][i][1] + 16*i, 8, mb_y>>1);



            dest_y += 16*s->linesize;

            dest_cb+= (16>>s->chroma_y_shift)*s->uvlinesize;

            dest_cr+= (16>>s->chroma_y_shift)*s->uvlinesize;

        }

        break;

    case MV_TYPE_DMV:

        if(s->picture_structure == PICT_FRAME){

            for(i=0; i<2; i++){

                int j;

                for(j=0; j<2; j++){

                    mpeg_motion_field(s, dest_y, dest_cb, dest_cr,

                                      j, j^i, ref_picture, pix_op,

                                      s->mv[dir][2*i + j][0],

                                      s->mv[dir][2*i + j][1], 8, mb_y);

                }

                pix_op = s->hdsp.avg_pixels_tab;

            }

        }else{

            for(i=0; i<2; i++){

                mpeg_motion(s, dest_y, dest_cb, dest_cr,

                            s->picture_structure != i+1,

                            ref_picture, pix_op,

                            s->mv[dir][2*i][0],s->mv[dir][2*i][1],16, mb_y>>1);



                // after put we make avg of the same block

                pix_op=s->hdsp.avg_pixels_tab;



                //opposite parity is always in the same frame if this is second field

                if(!s->first_field){

                    ref_picture = s->current_picture_ptr->f.data;

                }

            }

        }

    break;

    default: av_assert2(0);

    }

}
