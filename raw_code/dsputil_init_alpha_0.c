void dsputil_init_alpha(void)

{

    put_pixels_tab[0][0] = put_pixels16_axp_asm;

    put_pixels_tab[0][1] = put_pixels16_x2_axp;

    put_pixels_tab[0][2] = put_pixels16_y2_axp;

    put_pixels_tab[0][3] = put_pixels16_xy2_axp;



    put_no_rnd_pixels_tab[0][0] = put_pixels16_axp_asm;

    put_no_rnd_pixels_tab[0][1] = put_no_rnd_pixels16_x2_axp;

    put_no_rnd_pixels_tab[0][2] = put_no_rnd_pixels16_y2_axp;

    put_no_rnd_pixels_tab[0][3] = put_no_rnd_pixels16_xy2_axp;



    avg_pixels_tab[0][0] = avg_pixels16_axp;

    avg_pixels_tab[0][1] = avg_pixels16_x2_axp;

    avg_pixels_tab[0][2] = avg_pixels16_y2_axp;

    avg_pixels_tab[0][3] = avg_pixels16_xy2_axp;



    avg_no_rnd_pixels_tab[0][0] = avg_no_rnd_pixels16_axp;

    avg_no_rnd_pixels_tab[0][1] = avg_no_rnd_pixels16_x2_axp;

    avg_no_rnd_pixels_tab[0][2] = avg_no_rnd_pixels16_y2_axp;

    avg_no_rnd_pixels_tab[0][3] = avg_no_rnd_pixels16_xy2_axp;



    put_pixels_tab[1][0] = put_pixels_axp_asm;

    put_pixels_tab[1][1] = put_pixels_x2_axp;

    put_pixels_tab[1][2] = put_pixels_y2_axp;

    put_pixels_tab[1][3] = put_pixels_xy2_axp;



    put_no_rnd_pixels_tab[1][0] = put_pixels_axp_asm;

    put_no_rnd_pixels_tab[1][1] = put_no_rnd_pixels_x2_axp;

    put_no_rnd_pixels_tab[1][2] = put_no_rnd_pixels_y2_axp;

    put_no_rnd_pixels_tab[1][3] = put_no_rnd_pixels_xy2_axp;



    avg_pixels_tab[1][0] = avg_pixels_axp;

    avg_pixels_tab[1][1] = avg_pixels_x2_axp;

    avg_pixels_tab[1][2] = avg_pixels_y2_axp;

    avg_pixels_tab[1][3] = avg_pixels_xy2_axp;



    avg_no_rnd_pixels_tab[1][0] = avg_no_rnd_pixels_axp;

    avg_no_rnd_pixels_tab[1][1] = avg_no_rnd_pixels_x2_axp;

    avg_no_rnd_pixels_tab[1][2] = avg_no_rnd_pixels_y2_axp;

    avg_no_rnd_pixels_tab[1][3] = avg_no_rnd_pixels_xy2_axp;



    clear_blocks = clear_blocks_axp;



    /* amask clears all bits that correspond to present features.  */

    if (amask(AMASK_MVI) == 0) {

        put_pixels_clamped = put_pixels_clamped_mvi_asm;

        add_pixels_clamped = add_pixels_clamped_mvi_asm;



        get_pixels       = get_pixels_mvi;

        diff_pixels      = diff_pixels_mvi;

        pix_abs8x8       = pix_abs8x8_mvi;

        pix_abs16x16     = pix_abs16x16_mvi_asm;

        pix_abs16x16_x2  = pix_abs16x16_x2_mvi;

        pix_abs16x16_y2  = pix_abs16x16_y2_mvi;

        pix_abs16x16_xy2 = pix_abs16x16_xy2_mvi;

    }

}
