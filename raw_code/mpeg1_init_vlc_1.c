void mpeg1_init_vlc(MpegEncContext *s)

{

    static int done = 0;



    if (!done) {




        init_vlc(&dc_lum_vlc, 9, 12, 

                 vlc_dc_lum_bits, 1, 1,

                 vlc_dc_lum_code, 2, 2);

        init_vlc(&dc_chroma_vlc, 9, 12, 

                 vlc_dc_chroma_bits, 1, 1,

                 vlc_dc_chroma_code, 2, 2);

        init_vlc(&mv_vlc, 9, 17, 

                 &mbMotionVectorTable[0][1], 2, 1,

                 &mbMotionVectorTable[0][0], 2, 1);

        init_vlc(&mbincr_vlc, 9, 35, 

                 &mbAddrIncrTable[0][1], 2, 1,

                 &mbAddrIncrTable[0][0], 2, 1);

        init_vlc(&mb_pat_vlc, 9, 63, 

                 &mbPatTable[0][1], 2, 1,

                 &mbPatTable[0][0], 2, 1);

        

        init_vlc(&mb_ptype_vlc, 6, 32, 

                 &table_mb_ptype[0][1], 2, 1,

                 &table_mb_ptype[0][0], 2, 1);

        init_vlc(&mb_btype_vlc, 6, 32, 

                 &table_mb_btype[0][1], 2, 1,

                 &table_mb_btype[0][0], 2, 1);

        init_rl(&rl_mpeg1);

        init_rl(&rl_mpeg2);

        /* cannot use generic init because we must add the EOB code */

        init_vlc(&rl_mpeg1.vlc, 9, rl_mpeg1.n + 2, 

                 &rl_mpeg1.table_vlc[0][1], 4, 2,

                 &rl_mpeg1.table_vlc[0][0], 4, 2);

        init_vlc(&rl_mpeg2.vlc, 9, rl_mpeg2.n + 2, 

                 &rl_mpeg2.table_vlc[0][1], 4, 2,

                 &rl_mpeg2.table_vlc[0][0], 4, 2);

    }

}