static void flush_change(H264Context *h)

{

    h->outputed_poc = h->next_outputed_poc = INT_MIN;

    h->prev_interlaced_frame = 1;

    idr(h);

    h->prev_frame_num = -1;

    if (h->s.current_picture_ptr)

        h->s.current_picture_ptr->f.reference = 0;

    h->s.first_field = 0;

    memset(h->ref_list[0], 0, sizeof(h->ref_list[0]));

    memset(h->ref_list[1], 0, sizeof(h->ref_list[1]));

    memset(h->default_ref_list[0], 0, sizeof(h->default_ref_list[0]));

    memset(h->default_ref_list[1], 0, sizeof(h->default_ref_list[1]));

    ff_h264_reset_sei(h);

    h->recovery_frame= -1;

    h->sync= 0;

    h->list_count = 0;

    h->current_slice = 0;

}
