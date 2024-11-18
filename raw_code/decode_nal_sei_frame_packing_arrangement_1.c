static void decode_nal_sei_frame_packing_arrangement(HEVCContext *s)

{

    GetBitContext *gb = &s->HEVClc->gb;

    int cancel, type, quincunx, content;



    get_ue_golomb(gb);                  // frame_packing_arrangement_id

    cancel = get_bits1(gb);             // frame_packing_cancel_flag

    if (cancel == 0) {

        type     = get_bits(gb, 7);     // frame_packing_arrangement_type

        quincunx = get_bits1(gb);       // quincunx_sampling_flag

        content  = get_bits(gb, 6);     // content_interpretation_type



        // the following skips spatial_flipping_flag frame0_flipped_flag

        // field_views_flag current_frame_is_frame0_flag

        // frame0_self_contained_flag frame1_self_contained_flag

        skip_bits(gb, 6);



        if (quincunx == 0 && type != 5)

            skip_bits(gb, 16);  // frame[01]_grid_position_[xy]

        skip_bits(gb, 8);       // frame_packing_arrangement_reserved_byte

        skip_bits1(gb);         // frame_packing_arrangement_persistance_flag

    }

    skip_bits1(gb);             // upsampled_aspect_ratio_flag



    s->sei_frame_packing_present      = (cancel == 0);

    s->frame_packing_arrangement_type = type;

    s->content_interpretation_type    = content;

    s->quincunx_subsampling           = quincunx;

}
