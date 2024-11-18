static void sbr_mapping(AACContext *ac, SpectralBandReplication *sbr,

                        SBRData *ch_data, int e_a[2])

{

    int e, i, m;



    memset(ch_data->s_indexmapped[1], 0, 7*sizeof(ch_data->s_indexmapped[1]));

    for (e = 0; e < ch_data->bs_num_env; e++) {

        const unsigned int ilim = sbr->n[ch_data->bs_freq_res[e + 1]];

        uint16_t *table = ch_data->bs_freq_res[e + 1] ? sbr->f_tablehigh : sbr->f_tablelow;

        int k;



        for (i = 0; i < ilim; i++)

            for (m = table[i]; m < table[i + 1]; m++)

                sbr->e_origmapped[e][m - sbr->kx[1]] = ch_data->env_facs[e+1][i];



        // ch_data->bs_num_noise > 1 => 2 noise floors

        k = (ch_data->bs_num_noise > 1) && (ch_data->t_env[e] >= ch_data->t_q[1]);

        for (i = 0; i < sbr->n_q; i++)

            for (m = sbr->f_tablenoise[i]; m < sbr->f_tablenoise[i + 1]; m++)

                sbr->q_mapped[e][m - sbr->kx[1]] = ch_data->noise_facs[k+1][i];



        for (i = 0; i < sbr->n[1]; i++) {

            if (ch_data->bs_add_harmonic_flag) {

                const unsigned int m_midpoint =

                    (sbr->f_tablehigh[i] + sbr->f_tablehigh[i + 1]) >> 1;



                ch_data->s_indexmapped[e + 1][m_midpoint - sbr->kx[1]] = ch_data->bs_add_harmonic[i] *

                    (e >= e_a[1] || (ch_data->s_indexmapped[0][m_midpoint - sbr->kx[1]] == 1));

            }

        }



        for (i = 0; i < ilim; i++) {

            int additional_sinusoid_present = 0;

            for (m = table[i]; m < table[i + 1]; m++) {

                if (ch_data->s_indexmapped[e + 1][m - sbr->kx[1]]) {

                    additional_sinusoid_present = 1;

                    break;

                }

            }

            memset(&sbr->s_mapped[e][table[i] - sbr->kx[1]], additional_sinusoid_present,

                   (table[i + 1] - table[i]) * sizeof(sbr->s_mapped[e][0]));

        }

    }



    memcpy(ch_data->s_indexmapped[0], ch_data->s_indexmapped[ch_data->bs_num_env], sizeof(ch_data->s_indexmapped[0]));

}
