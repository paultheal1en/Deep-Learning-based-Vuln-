static void mpeg_decode_picture_coding_extension(MpegEncContext *s)

{

    s->full_pel[0] = s->full_pel[1] = 0;

    s->mpeg_f_code[0][0] = get_bits(&s->gb, 4);

    s->mpeg_f_code[0][1] = get_bits(&s->gb, 4);

    s->mpeg_f_code[1][0] = get_bits(&s->gb, 4);

    s->mpeg_f_code[1][1] = get_bits(&s->gb, 4);

    s->intra_dc_precision = get_bits(&s->gb, 2);

    s->picture_structure = get_bits(&s->gb, 2);

    s->top_field_first = get_bits1(&s->gb);

    s->frame_pred_frame_dct = get_bits1(&s->gb);

    s->concealment_motion_vectors = get_bits1(&s->gb);

    s->q_scale_type = get_bits1(&s->gb);

    s->intra_vlc_format = get_bits1(&s->gb);

    s->alternate_scan = get_bits1(&s->gb);

    s->repeat_first_field = get_bits1(&s->gb);

    s->chroma_420_type = get_bits1(&s->gb);

    s->progressive_frame = get_bits1(&s->gb);

    /* composite display not parsed */

    dprintf("intra_dc_precion=%d\n", s->intra_dc_precision);

    dprintf("picture_structure=%d\n", s->picture_structure);

    dprintf("conceal=%d\n", s->concealment_motion_vectors);

    dprintf("intra_vlc_format=%d\n", s->intra_vlc_format);

    dprintf("alternate_scan=%d\n", s->alternate_scan);

    dprintf("frame_pred_frame_dct=%d\n", s->frame_pred_frame_dct);

}
