int ff_h263_decode_mb(MpegEncContext *s,
                      int16_t block[6][64])
{
    int cbpc, cbpy, i, cbp, pred_x, pred_y, mx, my, dquant;
    int16_t *mot_val;
    const int xy= s->mb_x + s->mb_y * s->mb_stride;
    int cbpb = 0, pb_mv_count = 0;
    av_assert2(!s->h263_pred);
    if (s->pict_type == AV_PICTURE_TYPE_P) {
        do{
            if (get_bits1(&s->gb)) {
                /* skip mb */
                s->mb_intra = 0;
                for(i=0;i<6;i++)
                    s->block_last_index[i] = -1;
                s->mv_dir = MV_DIR_FORWARD;
                s->mv_type = MV_TYPE_16X16;
                s->current_picture.mb_type[xy] = MB_TYPE_SKIP | MB_TYPE_16x16 | MB_TYPE_L0;
                s->mv[0][0][0] = 0;
                s->mv[0][0][1] = 0;
                s->mb_skipped = !(s->obmc | s->loop_filter);
                goto end;
            cbpc = get_vlc2(&s->gb, ff_h263_inter_MCBPC_vlc.table, INTER_MCBPC_VLC_BITS, 2);
            if (cbpc < 0){
                av_log(s->avctx, AV_LOG_ERROR, "cbpc damaged at %d %d\n", s->mb_x, s->mb_y);
        }while(cbpc == 20);
        s->bdsp.clear_blocks(s->block[0]);
        dquant = cbpc & 8;
        s->mb_intra = ((cbpc & 4) != 0);
        if (s->mb_intra) goto intra;
        if(s->pb_frame && get_bits1(&s->gb))
            pb_mv_count = h263_get_modb(&s->gb, s->pb_frame, &cbpb);
        cbpy = get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1);
        if(s->alt_inter_vlc==0 || (cbpc & 3)!=3)
            cbpy ^= 0xF;
        cbp = (cbpc & 3) | (cbpy << 2);
        if (dquant) {
            h263_decode_dquant(s);
        s->mv_dir = MV_DIR_FORWARD;
        if ((cbpc & 16) == 0) {
            s->current_picture.mb_type[xy] = MB_TYPE_16x16 | MB_TYPE_L0;
            /* 16x16 motion prediction */
            s->mv_type = MV_TYPE_16X16;
            ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);
            if (s->umvplus)
               mx = h263p_decode_umotion(s, pred_x);
            else
               mx = ff_h263_decode_motion(s, pred_x, 1);
            if (mx >= 0xffff)
            if (s->umvplus)
               my = h263p_decode_umotion(s, pred_y);
            else
               my = ff_h263_decode_motion(s, pred_y, 1);
            if (my >= 0xffff)
            s->mv[0][0][0] = mx;
            s->mv[0][0][1] = my;
            if (s->umvplus && (mx - pred_x) == 1 && (my - pred_y) == 1)
               skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */
        } else {
            s->current_picture.mb_type[xy] = MB_TYPE_8x8 | MB_TYPE_L0;
            s->mv_type = MV_TYPE_8X8;
            for(i=0;i<4;i++) {
                mot_val = ff_h263_pred_motion(s, i, 0, &pred_x, &pred_y);
                if (s->umvplus)
                    mx = h263p_decode_umotion(s, pred_x);
                else
                    mx = ff_h263_decode_motion(s, pred_x, 1);
                if (mx >= 0xffff)
                if (s->umvplus)
                    my = h263p_decode_umotion(s, pred_y);
                else
                    my = ff_h263_decode_motion(s, pred_y, 1);
                if (my >= 0xffff)
                s->mv[0][i][0] = mx;
                s->mv[0][i][1] = my;
                if (s->umvplus && (mx - pred_x) == 1 && (my - pred_y) == 1)
                  skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */
                mot_val[0] = mx;
                mot_val[1] = my;
    } else if(s->pict_type==AV_PICTURE_TYPE_B) {
        int mb_type;
        const int stride= s->b8_stride;
        int16_t *mot_val0 = s->current_picture.motion_val[0][2 * (s->mb_x + s->mb_y * stride)];
        int16_t *mot_val1 = s->current_picture.motion_val[1][2 * (s->mb_x + s->mb_y * stride)];
//        const int mv_xy= s->mb_x + 1 + s->mb_y * s->mb_stride;
        //FIXME ugly
        mot_val0[0       ]= mot_val0[2       ]= mot_val0[0+2*stride]= mot_val0[2+2*stride]=
        mot_val0[1       ]= mot_val0[3       ]= mot_val0[1+2*stride]= mot_val0[3+2*stride]=
        mot_val1[0       ]= mot_val1[2       ]= mot_val1[0+2*stride]= mot_val1[2+2*stride]=
        mot_val1[1       ]= mot_val1[3       ]= mot_val1[1+2*stride]= mot_val1[3+2*stride]= 0;
        do{
            mb_type= get_vlc2(&s->gb, h263_mbtype_b_vlc.table, H263_MBTYPE_B_VLC_BITS, 2);
            if (mb_type < 0){
                av_log(s->avctx, AV_LOG_ERROR, "b mb_type damaged at %d %d\n", s->mb_x, s->mb_y);
            mb_type= h263_mb_type_b_map[ mb_type ];
        }while(!mb_type);
        s->mb_intra = IS_INTRA(mb_type);
        if(HAS_CBP(mb_type)){
            s->bdsp.clear_blocks(s->block[0]);
            cbpc = get_vlc2(&s->gb, cbpc_b_vlc.table, CBPC_B_VLC_BITS, 1);
            if(s->mb_intra){
                dquant = IS_QUANT(mb_type);
                goto intra;
            cbpy = get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1);
            if (cbpy < 0){
                av_log(s->avctx, AV_LOG_ERROR, "b cbpy damaged at %d %d\n", s->mb_x, s->mb_y);
            if(s->alt_inter_vlc==0 || (cbpc & 3)!=3)
                cbpy ^= 0xF;
            cbp = (cbpc & 3) | (cbpy << 2);
        }else
            cbp=0;
        av_assert2(!s->mb_intra);
        if(IS_QUANT(mb_type)){
            h263_decode_dquant(s);
        if(IS_DIRECT(mb_type)){
            s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;
            mb_type |= set_direct_mv(s);
        }else{
            s->mv_dir = 0;
            s->mv_type= MV_TYPE_16X16;
//FIXME UMV
            if(USES_LIST(mb_type, 0)){
                int16_t *mot_val= ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);
                s->mv_dir = MV_DIR_FORWARD;
                if (s->umvplus)
                    mx = h263p_decode_umotion(s, pred_x);
                else
                    mx = ff_h263_decode_motion(s, pred_x, 1);
                if (mx >= 0xffff)
                if (s->umvplus)
                    my = h263p_decode_umotion(s, pred_y);
                else
                    my = ff_h263_decode_motion(s, pred_y, 1);
                if (my >= 0xffff)
                if (s->umvplus && (mx - pred_x) == 1 && (my - pred_y) == 1)
                    skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */
                s->mv[0][0][0] = mx;
                s->mv[0][0][1] = my;
                mot_val[0       ]= mot_val[2       ]= mot_val[0+2*stride]= mot_val[2+2*stride]= mx;
                mot_val[1       ]= mot_val[3       ]= mot_val[1+2*stride]= mot_val[3+2*stride]= my;
            if(USES_LIST(mb_type, 1)){
                int16_t *mot_val= ff_h263_pred_motion(s, 0, 1, &pred_x, &pred_y);
                s->mv_dir |= MV_DIR_BACKWARD;
                if (s->umvplus)
                    mx = h263p_decode_umotion(s, pred_x);
                else
                    mx = ff_h263_decode_motion(s, pred_x, 1);
                if (mx >= 0xffff)
                if (s->umvplus)
                    my = h263p_decode_umotion(s, pred_y);
                else
                    my = ff_h263_decode_motion(s, pred_y, 1);
                if (my >= 0xffff)
                if (s->umvplus && (mx - pred_x) == 1 && (my - pred_y) == 1)
                    skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */
                s->mv[1][0][0] = mx;
                s->mv[1][0][1] = my;
                mot_val[0       ]= mot_val[2       ]= mot_val[0+2*stride]= mot_val[2+2*stride]= mx;
                mot_val[1       ]= mot_val[3       ]= mot_val[1+2*stride]= mot_val[3+2*stride]= my;
        s->current_picture.mb_type[xy] = mb_type;
    } else { /* I-Frame */
        do{
            cbpc = get_vlc2(&s->gb, ff_h263_intra_MCBPC_vlc.table, INTRA_MCBPC_VLC_BITS, 2);
            if (cbpc < 0){
                av_log(s->avctx, AV_LOG_ERROR, "I cbpc damaged at %d %d\n", s->mb_x, s->mb_y);
        }while(cbpc == 8);
        s->bdsp.clear_blocks(s->block[0]);
        dquant = cbpc & 4;
        s->mb_intra = 1;
intra:
        s->current_picture.mb_type[xy] = MB_TYPE_INTRA;
        if (s->h263_aic) {
            s->ac_pred = get_bits1(&s->gb);
            if(s->ac_pred){
                s->current_picture.mb_type[xy] = MB_TYPE_INTRA | MB_TYPE_ACPRED;
                s->h263_aic_dir = get_bits1(&s->gb);
        }else
            s->ac_pred = 0;
        if(s->pb_frame && get_bits1(&s->gb))
            pb_mv_count = h263_get_modb(&s->gb, s->pb_frame, &cbpb);
        cbpy = get_vlc2(&s->gb, ff_h263_cbpy_vlc.table, CBPY_VLC_BITS, 1);
        if(cbpy<0){
            av_log(s->avctx, AV_LOG_ERROR, "I cbpy damaged at %d %d\n", s->mb_x, s->mb_y);
        cbp = (cbpc & 3) | (cbpy << 2);
        if (dquant) {
            h263_decode_dquant(s);
        pb_mv_count += !!s->pb_frame;
    while(pb_mv_count--){
        ff_h263_decode_motion(s, 0, 1);
        ff_h263_decode_motion(s, 0, 1);
    /* decode each block */
    for (i = 0; i < 6; i++) {
        if (h263_decode_block(s, block[i], i, cbp&32) < 0)
            return -1;
        cbp+=cbp;
    if(s->pb_frame && h263_skip_b_part(s, cbpb) < 0)
        return -1;
    if(s->obmc && !s->mb_intra){
        if(s->pict_type == AV_PICTURE_TYPE_P && s->mb_x+1<s->mb_width && s->mb_num_left != 1)
            preview_obmc(s);
end:
        /* per-MB end of slice check */
    {
        int v= show_bits(&s->gb, 16);
        if (get_bits_left(&s->gb) < 16) {
            v >>= 16 - get_bits_left(&s->gb);
        if(v==0)
            return SLICE_END;
    return SLICE_OK;