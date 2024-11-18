static int mpeg4_decode_partitioned_mb(MpegEncContext *s, DCTELEM block[6][64])

{

    int cbp, mb_type;

    const int xy= s->mb_x + s->mb_y*s->mb_width;



    mb_type= s->mb_type[xy];

    cbp = s->cbp_table[xy];



    if(s->current_picture.qscale_table[xy] != s->qscale){

        s->qscale= s->current_picture.qscale_table[xy];

        s->y_dc_scale= s->y_dc_scale_table[ s->qscale ];

        s->c_dc_scale= s->c_dc_scale_table[ s->qscale ];

    }

    

    if (s->pict_type == P_TYPE || s->pict_type==S_TYPE) {

        int i;

        for(i=0; i<4; i++){

            s->mv[0][i][0] = s->motion_val[ s->block_index[i] ][0];

            s->mv[0][i][1] = s->motion_val[ s->block_index[i] ][1];

        }

        s->mb_intra = mb_type&MB_TYPE_INTRA;



        if (mb_type&MB_TYPE_SKIPED) {

            /* skip mb */

            for(i=0;i<6;i++)

                s->block_last_index[i] = -1;

            s->mv_dir = MV_DIR_FORWARD;

            s->mv_type = MV_TYPE_16X16;

            if(s->pict_type==S_TYPE && s->vol_sprite_usage==GMC_SPRITE){

                s->mcsel=1;

                s->mb_skiped = 0;

            }else{

                s->mcsel=0;

                s->mb_skiped = 1;

            }

        }else if(s->mb_intra){

            s->ac_pred = s->pred_dir_table[xy]>>7;



            /* decode each block */

            for (i = 0; i < 6; i++) {

                if(mpeg4_decode_block(s, block[i], i, cbp&32, 1) < 0){

                    fprintf(stderr, "texture corrupted at %d %d\n", s->mb_x, s->mb_y);

                    return -1;

                }

                cbp+=cbp;

            }

        }else if(!s->mb_intra){

//            s->mcsel= 0; //FIXME do we need to init that

            

            s->mv_dir = MV_DIR_FORWARD;

            if (mb_type&MB_TYPE_INTER4V) {

                s->mv_type = MV_TYPE_8X8;

            } else {

                s->mv_type = MV_TYPE_16X16;

            }

            /* decode each block */

            for (i = 0; i < 6; i++) {

                if(mpeg4_decode_block(s, block[i], i, cbp&32, 0) < 0){

                    fprintf(stderr, "texture corrupted at %d %d (trying to continue with mc/dc only)\n", s->mb_x, s->mb_y);

                    return -1;

                }

                cbp+=cbp;

            }

        }

    } else { /* I-Frame */

        int i;

        s->mb_intra = 1;

        s->ac_pred = s->pred_dir_table[xy]>>7;

        

        /* decode each block */

        for (i = 0; i < 6; i++) {

            if(mpeg4_decode_block(s, block[i], i, cbp&32, 1) < 0){

                fprintf(stderr, "texture corrupted at %d %d (trying to continue with dc only)\n", s->mb_x, s->mb_y);

                return -1;

            }

            cbp+=cbp;

        }

    }



    s->error_status_table[xy]&= ~AC_ERROR;



    /* per-MB end of slice check */



    if(--s->mb_num_left <= 0){

//printf("%06X %d\n", show_bits(&s->gb, 24), s->gb.size*8 - get_bits_count(&s->gb));

        if(mpeg4_is_resync(s))

            return SLICE_END;

        else

            return SLICE_NOEND;     

    }else{

        if(s->cbp_table[xy+1] && mpeg4_is_resync(s))

            return SLICE_END;

        else

            return SLICE_OK;

    }

}
