static int read_sbr_single_channel_element(AACContext *ac,

                                            SpectralBandReplication *sbr,

                                            GetBitContext *gb)

{

    int ret;



    if (get_bits1(gb)) // bs_data_extra

        skip_bits(gb, 4); // bs_reserved



    if (read_sbr_grid(ac, sbr, gb, &sbr->data[0]))

        return -1;

    read_sbr_dtdf(sbr, gb, &sbr->data[0]);

    read_sbr_invf(sbr, gb, &sbr->data[0]);

    read_sbr_envelope(sbr, gb, &sbr->data[0], 0);

    if((ret = read_sbr_noise(ac, sbr, gb, &sbr->data[0], 0)) < 0)

        return ret;



    if ((sbr->data[0].bs_add_harmonic_flag = get_bits1(gb)))

        get_bits1_vector(gb, sbr->data[0].bs_add_harmonic, sbr->n[1]);



    return 0;

}
