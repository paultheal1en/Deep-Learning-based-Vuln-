static void ff_compute_band_indexes(MPADecodeContext *s, GranuleDef *g)

{

    if (g->block_type == 2) {

        if (g->switch_point) {

            /* if switched mode, we handle the 36 first samples as

                long blocks.  For 8000Hz, we handle the 72 first

                exponents as long blocks */

            if (s->sample_rate_index <= 2)

                g->long_end = 8;

            else

                g->long_end = 6;



            g->short_start = 2 + (s->sample_rate_index != 8);

        } else {

            g->long_end    = 0;

            g->short_start = 0;

        }

    } else {

        g->short_start = 13;

        g->long_end    = 22;

    }

}
