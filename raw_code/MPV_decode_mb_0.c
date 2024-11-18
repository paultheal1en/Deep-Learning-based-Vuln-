void MPV_decode_mb(MpegEncContext *s, DCTELEM block[6][64])

{

    int mb_x, mb_y;

    int dct_linesize, dct_offset;

    op_pixels_func *op_pix;

    qpel_mc_func *op_qpix;



    mb_x = s->mb_x;

    mb_y = s->mb_y;



#ifdef FF_POSTPROCESS

    quant_store[mb_y][mb_x]=s->qscale;

    //printf("[%02d][%02d] %d\n",mb_x,mb_y,s->qscale);

#endif



    /* update DC predictors for P macroblocks */

    if (!s->mb_intra) {

        if (s->h263_pred || s->h263_aic) {

          if(s->mbintra_table[mb_x + mb_y*s->mb_width])

          {

            int wrap, xy, v;

            s->mbintra_table[mb_x + mb_y*s->mb_width]=0;

            wrap = 2 * s->mb_width + 2;

            xy = 2 * mb_x + 1 +  (2 * mb_y + 1) * wrap;

            v = 1024;

            

            s->dc_val[0][xy] = v;

            s->dc_val[0][xy + 1] = v;

            s->dc_val[0][xy + wrap] = v;

            s->dc_val[0][xy + 1 + wrap] = v;

            /* ac pred */

            memset(s->ac_val[0][xy], 0, 16 * sizeof(INT16));

            memset(s->ac_val[0][xy + 1], 0, 16 * sizeof(INT16));

            memset(s->ac_val[0][xy + wrap], 0, 16 * sizeof(INT16));

            memset(s->ac_val[0][xy + 1 + wrap], 0, 16 * sizeof(INT16));

            if (s->h263_msmpeg4) {

                s->coded_block[xy] = 0;

                s->coded_block[xy + 1] = 0;

                s->coded_block[xy + wrap] = 0;

                s->coded_block[xy + 1 + wrap] = 0;

            }

            /* chroma */

            wrap = s->mb_width + 2;

            xy = mb_x + 1 + (mb_y + 1) * wrap;

            s->dc_val[1][xy] = v;

            s->dc_val[2][xy] = v;

            /* ac pred */

            memset(s->ac_val[1][xy], 0, 16 * sizeof(INT16));

            memset(s->ac_val[2][xy], 0, 16 * sizeof(INT16));

          }

        } else {

            s->last_dc[0] = 128 << s->intra_dc_precision;

            s->last_dc[1] = 128 << s->intra_dc_precision;

            s->last_dc[2] = 128 << s->intra_dc_precision;

        }

    }

    else if (s->h263_pred || s->h263_aic)

        s->mbintra_table[mb_x + mb_y*s->mb_width]=1;



    /* update motion predictor, not for B-frames as they need the motion_val from the last P/S-Frame */

    if (s->out_format == FMT_H263) { //FIXME move into h263.c if possible, format specific stuff shouldnt be here

      if(s->pict_type!=B_TYPE){

        int xy, wrap, motion_x, motion_y;

        

        wrap = 2 * s->mb_width + 2;

        xy = 2 * mb_x + 1 + (2 * mb_y + 1) * wrap;

        if (s->mb_intra) {

            motion_x = 0;

            motion_y = 0;

            goto motion_init;

        } else if (s->mv_type == MV_TYPE_16X16) {

            motion_x = s->mv[0][0][0];

            motion_y = s->mv[0][0][1];

        motion_init:

            /* no update if 8X8 because it has been done during parsing */

            s->motion_val[xy][0] = motion_x;

            s->motion_val[xy][1] = motion_y;

            s->motion_val[xy + 1][0] = motion_x;

            s->motion_val[xy + 1][1] = motion_y;

            s->motion_val[xy + wrap][0] = motion_x;

            s->motion_val[xy + wrap][1] = motion_y;

            s->motion_val[xy + 1 + wrap][0] = motion_x;

            s->motion_val[xy + 1 + wrap][1] = motion_y;

        }

      }

    }

    

    if (!(s->encoding && (s->intra_only || s->pict_type==B_TYPE))) {

        UINT8 *dest_y, *dest_cb, *dest_cr;

        UINT8 *mbskip_ptr;



        /* avoid copy if macroblock skipped in last frame too 

           dont touch it for B-frames as they need the skip info from the next p-frame */

        if (s->pict_type != B_TYPE) {

            mbskip_ptr = &s->mbskip_table[s->mb_y * s->mb_width + s->mb_x];

            if (s->mb_skiped) {

                s->mb_skiped = 0;

                /* if previous was skipped too, then nothing to do ! 

                   skip only during decoding as we might trash the buffers during encoding a bit */

                if (*mbskip_ptr != 0 && !s->encoding) 

                    goto the_end;

                *mbskip_ptr = 1; /* indicate that this time we skiped it */

            } else {

                *mbskip_ptr = 0; /* not skipped */

            }

        }



        dest_y = s->current_picture[0] + (mb_y * 16 * s->linesize) + mb_x * 16;

        dest_cb = s->current_picture[1] + (mb_y * 8 * (s->linesize >> 1)) + mb_x * 8;

        dest_cr = s->current_picture[2] + (mb_y * 8 * (s->linesize >> 1)) + mb_x * 8;



        if (s->interlaced_dct) {

            dct_linesize = s->linesize * 2;

            dct_offset = s->linesize;

        } else {

            dct_linesize = s->linesize;

            dct_offset = s->linesize * 8;

        }



        if (!s->mb_intra) {

            /* motion handling */

            if((s->flags&CODEC_FLAG_HQ) || (!s->encoding)){

                if ((!s->no_rounding) || s->pict_type==B_TYPE){                

                    op_pix = put_pixels_tab;

                    op_qpix= qpel_mc_rnd_tab;

                }else{

                    op_pix = put_no_rnd_pixels_tab;

                    op_qpix= qpel_mc_no_rnd_tab;

                }



                if (s->mv_dir & MV_DIR_FORWARD) {

                    MPV_motion(s, dest_y, dest_cb, dest_cr, 0, s->last_picture, op_pix, op_qpix);

                    if ((!s->no_rounding) || s->pict_type==B_TYPE)

                        op_pix = avg_pixels_tab;

                    else

                        op_pix = avg_no_rnd_pixels_tab;

                }

                if (s->mv_dir & MV_DIR_BACKWARD) {

                    MPV_motion(s, dest_y, dest_cb, dest_cr, 1, s->next_picture, op_pix, op_qpix);

                }

            }



            /* add dct residue */

            add_dct(s, block[0], 0, dest_y, dct_linesize);

            add_dct(s, block[1], 1, dest_y + 8, dct_linesize);

            add_dct(s, block[2], 2, dest_y + dct_offset, dct_linesize);

            add_dct(s, block[3], 3, dest_y + dct_offset + 8, dct_linesize);



            add_dct(s, block[4], 4, dest_cb, s->linesize >> 1);

            add_dct(s, block[5], 5, dest_cr, s->linesize >> 1);

        } else {

            /* dct only in intra block */

            put_dct(s, block[0], 0, dest_y, dct_linesize);

            put_dct(s, block[1], 1, dest_y + 8, dct_linesize);

            put_dct(s, block[2], 2, dest_y + dct_offset, dct_linesize);

            put_dct(s, block[3], 3, dest_y + dct_offset + 8, dct_linesize);



            put_dct(s, block[4], 4, dest_cb, s->linesize >> 1);

            put_dct(s, block[5], 5, dest_cr, s->linesize >> 1);

        }

    }

 the_end:

    emms_c(); //FIXME remove

}
