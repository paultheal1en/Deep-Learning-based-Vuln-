static int msmpeg4v12_decode_mb(MpegEncContext *s, int16_t block[6][64])

{

    int cbp, code, i;

    uint32_t * const mb_type_ptr = &s->current_picture.mb_type[s->mb_x + s->mb_y*s->mb_stride];



    if (s->pict_type == AV_PICTURE_TYPE_P) {

        if (s->use_skip_mb_code) {

            if (get_bits1(&s->gb)) {

                /* skip mb */

                s->mb_intra = 0;

                for(i=0;i<6;i++)

                    s->block_last_index[i] = -1;

                s->mv_dir = MV_DIR_FORWARD;

                s->mv_type = MV_TYPE_16X16;

                s->mv[0][0][0] = 0;

                s->mv[0][0][1] = 0;

                s->mb_skipped = 1;

                *mb_type_ptr = MB_TYPE_SKIP | MB_TYPE_L0 | MB_TYPE_16x16;

                return 0;

            }

        }



        if(s->msmpeg4_version==2)

            code = get_vlc2(&s->gb, v2_mb_type_vlc.table, V2_MB_TYPE_VLC_BITS, 1);

        else

            code = get_vlc2(&s->gb, ff_h263_inter_MCBPC_vlc.table, INTER_MCBPC_VLC_BITS, 2);

        if(code<0 || code>7){

            av_log(s->avctx, AV_LOG_ERROR, "cbpc %d invalid at %d %d\n", code, s->mb_x, s->mb_y);

            return -1;

        }



        s->mb_intra = code >>2;



        cbp = code & 0x3;

    } else {

        s->mb_intra = 1;

        if(s->msmpeg4_version==2)

            cbp= get_vlc2(&s->gb, v2_intra_cbpc_vlc.table, V2_INTRA_CBPC_VLC_BITS, 1);

        else

            cbp= get_vlc2(&s->gb, ff_h263_intra_MCBPC_vlc.table, INTRA_MCBPC_VLC_BITS, 1);

        if(cbp<0 || cbp>3){

            av_log(s->avctx, AV_LOG_ERROR, "cbpc %d invalid at %d %d\n", cbp, s->mb_x, s->mb_y);

            return -1;

        }

    }



    if (!s->mb_intra) {

        int mx, my, cbpy;



        cbpy= get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1);

        if(cbpy<0){

            av_log(s->avctx, AV_LOG_ERROR, "cbpy %d invalid at %d %d\n", cbp, s->mb_x, s->mb_y);

            return -1;

        }



        cbp|= cbpy<<2;

        if(s->msmpeg4_version==1 || (cbp&3) != 3) cbp^= 0x3C;



        ff_h263_pred_motion(s, 0, 0, &mx, &my);

        mx= msmpeg4v2_decode_motion(s, mx, 1);

        my= msmpeg4v2_decode_motion(s, my, 1);



        s->mv_dir = MV_DIR_FORWARD;

        s->mv_type = MV_TYPE_16X16;

        s->mv[0][0][0] = mx;

        s->mv[0][0][1] = my;

        *mb_type_ptr = MB_TYPE_L0 | MB_TYPE_16x16;

    } else {

        if(s->msmpeg4_version==2){

            s->ac_pred = get_bits1(&s->gb);

            cbp|= get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1)<<2; //FIXME check errors

        } else{

            s->ac_pred = 0;

            cbp|= get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1)<<2; //FIXME check errors

            if(s->pict_type==AV_PICTURE_TYPE_P) cbp^=0x3C;

        }

        *mb_type_ptr = MB_TYPE_INTRA;

    }



    s->bdsp.clear_blocks(s->block[0]);

    for (i = 0; i < 6; i++) {

        if (ff_msmpeg4_decode_block(s, block[i], i, (cbp >> (5 - i)) & 1, NULL) < 0)

        {

             av_log(s->avctx, AV_LOG_ERROR, "\nerror while decoding block: %d x %d (%d)\n", s->mb_x, s->mb_y, i);

             return -1;

        }

    }

    return 0;

}
