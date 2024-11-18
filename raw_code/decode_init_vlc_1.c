static av_cold void decode_init_vlc(void){

    static int done = 0;



    if (!done) {

        int i;

        done = 1;



        init_vlc(&chroma_dc_coeff_token_vlc, CHROMA_DC_COEFF_TOKEN_VLC_BITS, 4*5,

                 &chroma_dc_coeff_token_len [0], 1, 1,

                 &chroma_dc_coeff_token_bits[0], 1, 1, 1);



        for(i=0; i<4; i++){

            init_vlc(&coeff_token_vlc[i], COEFF_TOKEN_VLC_BITS, 4*17,

                     &coeff_token_len [i][0], 1, 1,

                     &coeff_token_bits[i][0], 1, 1, 1);

        }



        for(i=0; i<3; i++){

            init_vlc(&chroma_dc_total_zeros_vlc[i], CHROMA_DC_TOTAL_ZEROS_VLC_BITS, 4,

                     &chroma_dc_total_zeros_len [i][0], 1, 1,

                     &chroma_dc_total_zeros_bits[i][0], 1, 1, 1);

        }

        for(i=0; i<15; i++){

            init_vlc(&total_zeros_vlc[i], TOTAL_ZEROS_VLC_BITS, 16,

                     &total_zeros_len [i][0], 1, 1,

                     &total_zeros_bits[i][0], 1, 1, 1);

        }



        for(i=0; i<6; i++){

            init_vlc(&run_vlc[i], RUN_VLC_BITS, 7,

                     &run_len [i][0], 1, 1,

                     &run_bits[i][0], 1, 1, 1);

        }

        init_vlc(&run7_vlc, RUN7_VLC_BITS, 16,

                 &run_len [6][0], 1, 1,

                 &run_bits[6][0], 1, 1, 1);

    }

}
