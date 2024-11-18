void ff_vdpau_mpeg_picture_complete(MpegEncContext *s, const uint8_t *buf,
                                    int buf_size, int slice_count)
{
    struct vdpau_render_state *render, *last, *next;
    int i;
    render = (struct vdpau_render_state *)s->current_picture_ptr->data[0];
    assert(render);
    /* fill VdpPictureInfoMPEG1Or2 struct */
    render->info.mpeg.picture_structure          = s->picture_structure;
    render->info.mpeg.picture_coding_type        = s->pict_type;
    render->info.mpeg.intra_dc_precision         = s->intra_dc_precision;
    render->info.mpeg.frame_pred_frame_dct       = s->frame_pred_frame_dct;
    render->info.mpeg.concealment_motion_vectors = s->concealment_motion_vectors;
    render->info.mpeg.intra_vlc_format           = s->intra_vlc_format;
    render->info.mpeg.alternate_scan             = s->alternate_scan;
    render->info.mpeg.q_scale_type               = s->q_scale_type;
    render->info.mpeg.top_field_first            = s->top_field_first;
    render->info.mpeg.full_pel_forward_vector    = s->full_pel[0]; // MPEG-1 only.  Set 0 for MPEG-2
    render->info.mpeg.full_pel_backward_vector   = s->full_pel[1]; // MPEG-1 only.  Set 0 for MPEG-2
    render->info.mpeg.f_code[0][0]               = s->mpeg_f_code[0][0]; // For MPEG-1 fill both horiz. & vert.
    render->info.mpeg.f_code[0][1]               = s->mpeg_f_code[0][1];
    render->info.mpeg.f_code[1][0]               = s->mpeg_f_code[1][0];
    render->info.mpeg.f_code[1][1]               = s->mpeg_f_code[1][1];
    for (i = 0; i < 64; ++i) {
        render->info.mpeg.intra_quantizer_matrix[i]     = s->intra_matrix[i];
        render->info.mpeg.non_intra_quantizer_matrix[i] = s->inter_matrix[i];
    }
    render->info.mpeg.forward_reference          = VDP_INVALID_HANDLE;
    render->info.mpeg.backward_reference         = VDP_INVALID_HANDLE;
    switch(s->pict_type){
    case  FF_B_TYPE:
        next = (struct vdpau_render_state *)s->next_picture.data[0];
        assert(next);
        render->info.mpeg.backward_reference     = next->surface;
        // no return here, going to set forward prediction
    case  FF_P_TYPE:
        last = (struct vdpau_render_state *)s->last_picture.data[0];
        if (!last) // FIXME: Does this test make sense?
            last = render; // predict second field from the first
        render->info.mpeg.forward_reference      = last->surface;
    }
    ff_vdpau_add_data_chunk(s, buf, buf_size);
    render->info.mpeg.slice_count                = slice_count;
    if (slice_count)
        ff_draw_horiz_band(s, 0, s->avctx->height);
    render->bitstream_buffers_used               = 0;
}