av_cold int ff_msmpeg4_decode_init(AVCodecContext *avctx)
{
    MpegEncContext *s = avctx->priv_data;
    static volatile int done = 0;
    int i;
    MVTable *mv;
    if (ff_h263_decode_init(avctx) < 0)
    ff_msmpeg4_common_init(s);
    if (!done) {
        for(i=0;i<NB_RL_TABLES;i++) {
            ff_init_rl(&ff_rl_table[i], ff_static_rl_table_store[i]);
        INIT_VLC_RL(ff_rl_table[0], 642);
        INIT_VLC_RL(ff_rl_table[1], 1104);
        INIT_VLC_RL(ff_rl_table[2], 554);
        INIT_VLC_RL(ff_rl_table[3], 940);
        INIT_VLC_RL(ff_rl_table[4], 962);
        INIT_VLC_RL(ff_rl_table[5], 554);
        mv = &ff_mv_tables[0];
        INIT_VLC_STATIC(&mv->vlc, MV_VLC_BITS, mv->n + 1,
                    mv->table_mv_bits, 1, 1,
                    mv->table_mv_code, 2, 2, 3714);
        mv = &ff_mv_tables[1];
        INIT_VLC_STATIC(&mv->vlc, MV_VLC_BITS, mv->n + 1,
                    mv->table_mv_bits, 1, 1,
                    mv->table_mv_code, 2, 2, 2694);
        INIT_VLC_STATIC(&ff_msmp4_dc_luma_vlc[0], DC_VLC_BITS, 120,
                 &ff_table0_dc_lum[0][1], 8, 4,
                 &ff_table0_dc_lum[0][0], 8, 4, 1158);
        INIT_VLC_STATIC(&ff_msmp4_dc_chroma_vlc[0], DC_VLC_BITS, 120,
                 &ff_table0_dc_chroma[0][1], 8, 4,
                 &ff_table0_dc_chroma[0][0], 8, 4, 1118);
        INIT_VLC_STATIC(&ff_msmp4_dc_luma_vlc[1], DC_VLC_BITS, 120,
                 &ff_table1_dc_lum[0][1], 8, 4,
                 &ff_table1_dc_lum[0][0], 8, 4, 1476);
        INIT_VLC_STATIC(&ff_msmp4_dc_chroma_vlc[1], DC_VLC_BITS, 120,
                 &ff_table1_dc_chroma[0][1], 8, 4,
                 &ff_table1_dc_chroma[0][0], 8, 4, 1216);
        INIT_VLC_STATIC(&v2_dc_lum_vlc, DC_VLC_BITS, 512,
                 &ff_v2_dc_lum_table[0][1], 8, 4,
                 &ff_v2_dc_lum_table[0][0], 8, 4, 1472);
        INIT_VLC_STATIC(&v2_dc_chroma_vlc, DC_VLC_BITS, 512,
                 &ff_v2_dc_chroma_table[0][1], 8, 4,
                 &ff_v2_dc_chroma_table[0][0], 8, 4, 1506);
        INIT_VLC_STATIC(&v2_intra_cbpc_vlc, V2_INTRA_CBPC_VLC_BITS, 4,
                 &ff_v2_intra_cbpc[0][1], 2, 1,
                 &ff_v2_intra_cbpc[0][0], 2, 1, 8);
        INIT_VLC_STATIC(&v2_mb_type_vlc, V2_MB_TYPE_VLC_BITS, 8,
                 &ff_v2_mb_type[0][1], 2, 1,
                 &ff_v2_mb_type[0][0], 2, 1, 128);
        INIT_VLC_STATIC(&v2_mv_vlc, V2_MV_VLC_BITS, 33,
                 &ff_mvtab[0][1], 2, 1,
                 &ff_mvtab[0][0], 2, 1, 538);
        INIT_VLC_STATIC(&ff_mb_non_intra_vlc[0], MB_NON_INTRA_VLC_BITS, 128,
                     &ff_wmv2_inter_table[0][0][1], 8, 4,
                     &ff_wmv2_inter_table[0][0][0], 8, 4, 1636);
        INIT_VLC_STATIC(&ff_mb_non_intra_vlc[1], MB_NON_INTRA_VLC_BITS, 128,
                     &ff_wmv2_inter_table[1][0][1], 8, 4,
                     &ff_wmv2_inter_table[1][0][0], 8, 4, 2648);
        INIT_VLC_STATIC(&ff_mb_non_intra_vlc[2], MB_NON_INTRA_VLC_BITS, 128,
                     &ff_wmv2_inter_table[2][0][1], 8, 4,
                     &ff_wmv2_inter_table[2][0][0], 8, 4, 1532);
        INIT_VLC_STATIC(&ff_mb_non_intra_vlc[3], MB_NON_INTRA_VLC_BITS, 128,
                     &ff_wmv2_inter_table[3][0][1], 8, 4,
                     &ff_wmv2_inter_table[3][0][0], 8, 4, 2488);
        INIT_VLC_STATIC(&ff_msmp4_mb_i_vlc, MB_INTRA_VLC_BITS, 64,
                 &ff_msmp4_mb_i_table[0][1], 4, 2,
                 &ff_msmp4_mb_i_table[0][0], 4, 2, 536);
        INIT_VLC_STATIC(&ff_inter_intra_vlc, INTER_INTRA_VLC_BITS, 4,
                 &ff_table_inter_intra[0][1], 2, 1,
                 &ff_table_inter_intra[0][0], 2, 1, 8);
        done = 1;
    switch(s->msmpeg4_version){
    case 1:
    case 2:
        s->decode_mb= msmpeg4v12_decode_mb;
        break;
    case 3:
    case 4:
        s->decode_mb= msmpeg4v34_decode_mb;
        break;
    case 5:
        if (CONFIG_WMV2_DECODER)
            s->decode_mb= ff_wmv2_decode_mb;
    case 6:
        //FIXME + TODO VC1 decode mb
        break;
    s->slice_height= s->mb_height; //to avoid 1/0 if the first frame is not a keyframe
    return 0;