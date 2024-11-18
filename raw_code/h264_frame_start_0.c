static int h264_frame_start(H264Context *h)

{

    H264Picture *pic;

    int i, ret;

    const int pixel_shift = h->pixel_shift;



    ret = initialize_cur_frame(h);

    if (ret < 0)

        return ret;



    pic = h->cur_pic_ptr;

    pic->reference              = h->droppable ? 0 : h->picture_structure;

    pic->f->coded_picture_number = h->coded_picture_number++;

    pic->field_picture          = h->picture_structure != PICT_FRAME;

    pic->frame_num               = h->frame_num;

    /*

     * Zero key_frame here; IDR markings per slice in frame or fields are ORed

     * in later.

     * See decode_nal_units().

     */

    pic->f->key_frame = 0;

    pic->mmco_reset  = 0;

    pic->recovered   = 0;



    if (CONFIG_ERROR_RESILIENCE && h->enable_er)

        ff_er_frame_start(&h->slice_ctx[0].er);



    for (i = 0; i < 16; i++) {

        h->block_offset[i]           = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 4 * pic->f->linesize[0] * ((scan8[i] - scan8[0]) >> 3);

        h->block_offset[48 + i]      = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 8 * pic->f->linesize[0] * ((scan8[i] - scan8[0]) >> 3);

    }

    for (i = 0; i < 16; i++) {

        h->block_offset[16 + i]      =

        h->block_offset[32 + i]      = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 4 * pic->f->linesize[1] * ((scan8[i] - scan8[0]) >> 3);

        h->block_offset[48 + 16 + i] =

        h->block_offset[48 + 32 + i] = (4 * ((scan8[i] - scan8[0]) & 7) << pixel_shift) + 8 * pic->f->linesize[1] * ((scan8[i] - scan8[0]) >> 3);

    }



    /* Some macroblocks can be accessed before they're available in case

     * of lost slices, MBAFF or threading. */

    memset(h->slice_table, -1,

           (h->mb_height * h->mb_stride - 1) * sizeof(*h->slice_table));



    /* We mark the current picture as non-reference after allocating it, so

     * that if we break out due to an error it can be released automatically

     * in the next ff_mpv_frame_start().

     */

    h->cur_pic_ptr->reference = 0;



    h->cur_pic_ptr->field_poc[0] = h->cur_pic_ptr->field_poc[1] = INT_MAX;



    h->next_output_pic = NULL;



    assert(h->cur_pic_ptr->long_ref == 0);



    return 0;

}
