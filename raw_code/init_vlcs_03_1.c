static void init_vlcs(ASV1Context *a){

    static int done = 0;



    if (!done) {

        done = 1;



        init_vlc(&ccp_vlc, VLC_BITS, 17, 

                 &ccp_tab[0][1], 2, 1,

                 &ccp_tab[0][0], 2, 1);

        init_vlc(&dc_ccp_vlc, VLC_BITS, 8, 

                 &dc_ccp_tab[0][1], 2, 1,

                 &dc_ccp_tab[0][0], 2, 1);

        init_vlc(&ac_ccp_vlc, VLC_BITS, 16, 

                 &ac_ccp_tab[0][1], 2, 1,

                 &ac_ccp_tab[0][0], 2, 1);

        init_vlc(&level_vlc,  VLC_BITS, 7, 

                 &level_tab[0][1], 2, 1,

                 &level_tab[0][0], 2, 1);

        init_vlc(&asv2_level_vlc, ASV2_LEVEL_VLC_BITS, 63, 

                 &asv2_level_tab[0][1], 2, 1,

                 &asv2_level_tab[0][0], 2, 1);

    }

}
