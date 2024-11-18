av_cold void ff_dsputil_init_vis(DSPContext *c, AVCodecContext *avctx)

{

  /* VIS-specific optimizations */

  int accel = vis_level ();

  const int high_bit_depth = avctx->bits_per_raw_sample > 8;



  if (accel & ACCEL_SPARC_VIS) {

      if (avctx->bits_per_raw_sample <= 8 &&

          avctx->idct_algo == FF_IDCT_SIMPLEVIS) {

          c->idct_put = ff_simple_idct_put_vis;

          c->idct_add = ff_simple_idct_add_vis;

          c->idct     = ff_simple_idct_vis;

          c->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

      }



      if (!high_bit_depth) {

      c->put_pixels_tab[0][0] = MC_put_o_16_vis;

      c->put_pixels_tab[0][1] = MC_put_x_16_vis;

      c->put_pixels_tab[0][2] = MC_put_y_16_vis;

      c->put_pixels_tab[0][3] = MC_put_xy_16_vis;



      c->put_pixels_tab[1][0] = MC_put_o_8_vis;

      c->put_pixels_tab[1][1] = MC_put_x_8_vis;

      c->put_pixels_tab[1][2] = MC_put_y_8_vis;

      c->put_pixels_tab[1][3] = MC_put_xy_8_vis;



      c->avg_pixels_tab[0][0] = MC_avg_o_16_vis;

      c->avg_pixels_tab[0][1] = MC_avg_x_16_vis;

      c->avg_pixels_tab[0][2] = MC_avg_y_16_vis;

      c->avg_pixels_tab[0][3] = MC_avg_xy_16_vis;



      c->avg_pixels_tab[1][0] = MC_avg_o_8_vis;

      c->avg_pixels_tab[1][1] = MC_avg_x_8_vis;

      c->avg_pixels_tab[1][2] = MC_avg_y_8_vis;

      c->avg_pixels_tab[1][3] = MC_avg_xy_8_vis;



      c->put_no_rnd_pixels_tab[0][0] = MC_put_no_round_o_16_vis;

      c->put_no_rnd_pixels_tab[0][1] = MC_put_no_round_x_16_vis;

      c->put_no_rnd_pixels_tab[0][2] = MC_put_no_round_y_16_vis;

      c->put_no_rnd_pixels_tab[0][3] = MC_put_no_round_xy_16_vis;



      c->put_no_rnd_pixels_tab[1][0] = MC_put_no_round_o_8_vis;

      c->put_no_rnd_pixels_tab[1][1] = MC_put_no_round_x_8_vis;

      c->put_no_rnd_pixels_tab[1][2] = MC_put_no_round_y_8_vis;

      c->put_no_rnd_pixels_tab[1][3] = MC_put_no_round_xy_8_vis;



      c->avg_no_rnd_pixels_tab[0] = MC_avg_no_round_o_16_vis;

      c->avg_no_rnd_pixels_tab[1] = MC_avg_no_round_x_16_vis;

      c->avg_no_rnd_pixels_tab[2] = MC_avg_no_round_y_16_vis;

      c->avg_no_rnd_pixels_tab[3] = MC_avg_no_round_xy_16_vis;

      }

  }

}
