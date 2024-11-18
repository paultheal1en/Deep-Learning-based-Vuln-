static void hls_decode_neighbour(HEVCContext *s, int x_ctb, int y_ctb,

                                 int ctb_addr_ts)

{

    HEVCLocalContext *lc  = &s->HEVClc;

    int ctb_size          = 1 << s->ps.sps->log2_ctb_size;

    int ctb_addr_rs       = s->ps.pps->ctb_addr_ts_to_rs[ctb_addr_ts];

    int ctb_addr_in_slice = ctb_addr_rs - s->sh.slice_addr;



    s->tab_slice_address[ctb_addr_rs] = s->sh.slice_addr;



    if (s->ps.pps->entropy_coding_sync_enabled_flag) {

        if (x_ctb == 0 && (y_ctb & (ctb_size - 1)) == 0)

            lc->first_qp_group = 1;

        lc->end_of_tiles_x = s->ps.sps->width;

    } else if (s->ps.pps->tiles_enabled_flag) {

        if (ctb_addr_ts && s->ps.pps->tile_id[ctb_addr_ts] != s->ps.pps->tile_id[ctb_addr_ts - 1]) {

            int idxX = s->ps.pps->col_idxX[x_ctb >> s->ps.sps->log2_ctb_size];

            lc->start_of_tiles_x = x_ctb;

            lc->end_of_tiles_x   = x_ctb + (s->ps.pps->column_width[idxX] << s->ps.sps->log2_ctb_size);

            lc->first_qp_group   = 1;

        }

    } else {

        lc->end_of_tiles_x = s->ps.sps->width;

    }



    lc->end_of_tiles_y = FFMIN(y_ctb + ctb_size, s->ps.sps->height);



    lc->boundary_flags = 0;

    if (s->ps.pps->tiles_enabled_flag) {

        if (x_ctb > 0 && s->ps.pps->tile_id[ctb_addr_ts] != s->ps.pps->tile_id[s->ps.pps->ctb_addr_rs_to_ts[ctb_addr_rs - 1]])

            lc->boundary_flags |= BOUNDARY_LEFT_TILE;

        if (x_ctb > 0 && s->tab_slice_address[ctb_addr_rs] != s->tab_slice_address[ctb_addr_rs - 1])

            lc->boundary_flags |= BOUNDARY_LEFT_SLICE;

        if (y_ctb > 0 && s->ps.pps->tile_id[ctb_addr_ts] != s->ps.pps->tile_id[s->ps.pps->ctb_addr_rs_to_ts[ctb_addr_rs - s->ps.sps->ctb_width]])

            lc->boundary_flags |= BOUNDARY_UPPER_TILE;

        if (y_ctb > 0 && s->tab_slice_address[ctb_addr_rs] != s->tab_slice_address[ctb_addr_rs - s->ps.sps->ctb_width])

            lc->boundary_flags |= BOUNDARY_UPPER_SLICE;

    } else {

        if (!ctb_addr_in_slice > 0)

            lc->boundary_flags |= BOUNDARY_LEFT_SLICE;

        if (ctb_addr_in_slice < s->ps.sps->ctb_width)

            lc->boundary_flags |= BOUNDARY_UPPER_SLICE;

    }



    lc->ctb_left_flag = ((x_ctb > 0) && (ctb_addr_in_slice > 0) && !(lc->boundary_flags & BOUNDARY_LEFT_TILE));

    lc->ctb_up_flag   = ((y_ctb > 0) && (ctb_addr_in_slice >= s->ps.sps->ctb_width) && !(lc->boundary_flags & BOUNDARY_UPPER_TILE));

    lc->ctb_up_right_flag = ((y_ctb > 0)  && (ctb_addr_in_slice+1 >= s->ps.sps->ctb_width) && (s->ps.pps->tile_id[ctb_addr_ts] == s->ps.pps->tile_id[s->ps.pps->ctb_addr_rs_to_ts[ctb_addr_rs+1 - s->ps.sps->ctb_width]]));

    lc->ctb_up_left_flag = ((x_ctb > 0) && (y_ctb > 0)  && (ctb_addr_in_slice-1 >= s->ps.sps->ctb_width) && (s->ps.pps->tile_id[ctb_addr_ts] == s->ps.pps->tile_id[s->ps.pps->ctb_addr_rs_to_ts[ctb_addr_rs-1 - s->ps.sps->ctb_width]]));

}
