static void read_sbr_noise(SpectralBandReplication *sbr, GetBitContext *gb,

                           SBRData *ch_data, int ch)

{

    int i, j;

    VLC_TYPE (*t_huff)[2], (*f_huff)[2];

    int t_lav, f_lav;

    int delta = (ch == 1 && sbr->bs_coupling == 1) + 1;



    if (sbr->bs_coupling && ch) {

        t_huff = vlc_sbr[T_HUFFMAN_NOISE_BAL_3_0DB].table;

        t_lav  = vlc_sbr_lav[T_HUFFMAN_NOISE_BAL_3_0DB];

        f_huff = vlc_sbr[F_HUFFMAN_ENV_BAL_3_0DB].table;

        f_lav  = vlc_sbr_lav[F_HUFFMAN_ENV_BAL_3_0DB];

    } else {

        t_huff = vlc_sbr[T_HUFFMAN_NOISE_3_0DB].table;

        t_lav  = vlc_sbr_lav[T_HUFFMAN_NOISE_3_0DB];

        f_huff = vlc_sbr[F_HUFFMAN_ENV_3_0DB].table;

        f_lav  = vlc_sbr_lav[F_HUFFMAN_ENV_3_0DB];

    }



#if USE_FIXED

    for (i = 0; i < ch_data->bs_num_noise; i++) {

        if (ch_data->bs_df_noise[i]) {

            for (j = 0; j < sbr->n_q; j++)

                ch_data->noise_facs[i + 1][j].mant = ch_data->noise_facs[i][j].mant + delta * (get_vlc2(gb, t_huff, 9, 2) - t_lav);

        } else {

            ch_data->noise_facs[i + 1][0].mant = delta * get_bits(gb, 5); // bs_noise_start_value_balance or bs_noise_start_value_level

            for (j = 1; j < sbr->n_q; j++)

                ch_data->noise_facs[i + 1][j].mant = ch_data->noise_facs[i + 1][j - 1].mant + delta * (get_vlc2(gb, f_huff, 9, 3) - f_lav);

        }

    }

#else

    for (i = 0; i < ch_data->bs_num_noise; i++) {

        if (ch_data->bs_df_noise[i]) {

            for (j = 0; j < sbr->n_q; j++)

                ch_data->noise_facs[i + 1][j] = ch_data->noise_facs[i][j] + delta * (get_vlc2(gb, t_huff, 9, 2) - t_lav);

        } else {

            ch_data->noise_facs[i + 1][0] = delta * get_bits(gb, 5); // bs_noise_start_value_balance or bs_noise_start_value_level

            for (j = 1; j < sbr->n_q; j++)

                ch_data->noise_facs[i + 1][j] = ch_data->noise_facs[i + 1][j - 1] + delta * (get_vlc2(gb, f_huff, 9, 3) - f_lav);

        }

    }

#endif /* USE_FIXED */



    //assign 0th elements of noise_facs from last elements

    memcpy(ch_data->noise_facs[0], ch_data->noise_facs[ch_data->bs_num_noise],

           sizeof(ch_data->noise_facs[0]));

}
