static void h261_decode_init_vlc(H261Context *h){

    static int done = 0;



    if(!done){

        done = 1;

        init_vlc(&h261_mba_vlc, H261_MBA_VLC_BITS, 35,

                 h261_mba_bits, 1, 1,

                 h261_mba_code, 1, 1);

        init_vlc(&h261_mtype_vlc, H261_MTYPE_VLC_BITS, 10,

                 h261_mtype_bits, 1, 1,

                 h261_mtype_code, 1, 1);

        init_vlc(&h261_mv_vlc, H261_MV_VLC_BITS, 17,

                 &h261_mv_tab[0][1], 2, 1,

                 &h261_mv_tab[0][0], 2, 1);

        init_vlc(&h261_cbp_vlc, H261_CBP_VLC_BITS, 63,

                 &h261_cbp_tab[0][1], 2, 1,

                 &h261_cbp_tab[0][0], 2, 1);

        init_rl(&h261_rl_tcoeff);

        init_vlc_rl(&h261_rl_tcoeff);

    }

}
