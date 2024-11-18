static void unpack_dct_coeffs(Vp3DecodeContext *s, GetBitContext *gb)

{

    int i;

    int dc_y_table;

    int dc_c_table;

    int ac_y_table;

    int ac_c_table;

    int residual_eob_run = 0;



    /* fetch the DC table indices */

    dc_y_table = get_bits(gb, 4);

    dc_c_table = get_bits(gb, 4);



    /* unpack the Y plane DC coefficients */

    debug_vp3("  vp3: unpacking Y plane DC coefficients using table %d\n",

        dc_y_table);

    residual_eob_run = unpack_vlcs(s, gb, &s->dc_vlc[dc_y_table], 0, 

        s->first_coded_y_fragment, s->last_coded_y_fragment, residual_eob_run);



    /* unpack the C plane DC coefficients */

    debug_vp3("  vp3: unpacking C plane DC coefficients using table %d\n",

        dc_c_table);

    residual_eob_run = unpack_vlcs(s, gb, &s->dc_vlc[dc_c_table], 0,

        s->first_coded_c_fragment, s->last_coded_c_fragment, residual_eob_run);



    /* fetch the AC table indices */

    ac_y_table = get_bits(gb, 4);

    ac_c_table = get_bits(gb, 4);



    /* unpack the group 1 AC coefficients (coeffs 1-5) */

    for (i = 1; i <= 5; i++) {



        debug_vp3("  vp3: unpacking level %d Y plane AC coefficients using table %d\n",

            i, ac_y_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_1[ac_y_table], i, 

            s->first_coded_y_fragment, s->last_coded_y_fragment, residual_eob_run);



        debug_vp3("  vp3: unpacking level %d C plane AC coefficients using table %d\n",

            i, ac_c_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_1[ac_c_table], i, 

            s->first_coded_c_fragment, s->last_coded_c_fragment, residual_eob_run);

    }



    /* unpack the group 2 AC coefficients (coeffs 6-14) */

    for (i = 6; i <= 14; i++) {



        debug_vp3("  vp3: unpacking level %d Y plane AC coefficients using table %d\n",

            i, ac_y_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_2[ac_y_table], i, 

            s->first_coded_y_fragment, s->last_coded_y_fragment, residual_eob_run);



        debug_vp3("  vp3: unpacking level %d C plane AC coefficients using table %d\n",

            i, ac_c_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_2[ac_c_table], i, 

            s->first_coded_c_fragment, s->last_coded_c_fragment, residual_eob_run);

    }



    /* unpack the group 3 AC coefficients (coeffs 15-27) */

    for (i = 15; i <= 27; i++) {



        debug_vp3("  vp3: unpacking level %d Y plane AC coefficients using table %d\n",

            i, ac_y_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_3[ac_y_table], i, 

            s->first_coded_y_fragment, s->last_coded_y_fragment, residual_eob_run);



        debug_vp3("  vp3: unpacking level %d C plane AC coefficients using table %d\n",

            i, ac_c_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_3[ac_c_table], i, 

            s->first_coded_c_fragment, s->last_coded_c_fragment, residual_eob_run);

    }



    /* unpack the group 4 AC coefficients (coeffs 28-63) */

    for (i = 28; i <= 63; i++) {



        debug_vp3("  vp3: unpacking level %d Y plane AC coefficients using table %d\n",

            i, ac_y_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_4[ac_y_table], i, 

            s->first_coded_y_fragment, s->last_coded_y_fragment, residual_eob_run);



        debug_vp3("  vp3: unpacking level %d C plane AC coefficients using table %d\n",

            i, ac_c_table);

        residual_eob_run = unpack_vlcs(s, gb, &s->ac_vlc_4[ac_c_table], i, 

            s->first_coded_c_fragment, s->last_coded_c_fragment, residual_eob_run);

    }

}
