void ff_hevc_cabac_init(HEVCContext *s, int ctb_addr_ts)

{

    if (ctb_addr_ts == s->ps.pps->ctb_addr_rs_to_ts[s->sh.slice_ctb_addr_rs]) {

        cabac_init_decoder(s);

        if (s->sh.dependent_slice_segment_flag == 0 ||

            (s->ps.pps->tiles_enabled_flag &&

             s->ps.pps->tile_id[ctb_addr_ts] != s->ps.pps->tile_id[ctb_addr_ts - 1]))

            cabac_init_state(s);



        if (!s->sh.first_slice_in_pic_flag &&

            s->ps.pps->entropy_coding_sync_enabled_flag) {

            if (ctb_addr_ts % s->ps.sps->ctb_width == 0) {

                if (s->ps.sps->ctb_width == 1)

                    cabac_init_state(s);

                else if (s->sh.dependent_slice_segment_flag == 1)

                    load_states(s);

            }

        }

    } else {

        if (s->ps.pps->tiles_enabled_flag &&

            s->ps.pps->tile_id[ctb_addr_ts] != s->ps.pps->tile_id[ctb_addr_ts - 1]) {

            if (s->threads_number == 1)

                cabac_reinit(s->HEVClc);

            else

                cabac_init_decoder(s);

            cabac_init_state(s);

        }

        if (s->ps.pps->entropy_coding_sync_enabled_flag) {

            if (ctb_addr_ts % s->ps.sps->ctb_width == 0) {

                get_cabac_terminate(&s->HEVClc->cc);

                if (s->threads_number == 1)

                    cabac_reinit(s->HEVClc);

                else

                    cabac_init_decoder(s);



                if (s->ps.sps->ctb_width == 1)

                    cabac_init_state(s);

                else

                    load_states(s);

            }

        }

    }

}
