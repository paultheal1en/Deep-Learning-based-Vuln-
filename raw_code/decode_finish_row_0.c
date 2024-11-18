static void decode_finish_row(H264Context *h, H264SliceContext *sl)

{

    int top            = 16 * (h->mb_y      >> FIELD_PICTURE(h));

    int pic_height     = 16 *  h->mb_height >> FIELD_PICTURE(h);

    int height         =  16      << FRAME_MBAFF(h);

    int deblock_border = (16 + 4) << FRAME_MBAFF(h);



    if (h->deblocking_filter) {

        if ((top + height) >= pic_height)

            height += deblock_border;

        top -= deblock_border;

    }



    if (top >= pic_height || (top + height) < 0)

        return;



    height = FFMIN(height, pic_height - top);

    if (top < 0) {

        height = top + height;

        top    = 0;

    }



    ff_h264_draw_horiz_band(h, sl, top, height);



    if (h->droppable)

        return;



    ff_thread_report_progress(&h->cur_pic_ptr->tf, top + height - 1,

                              h->picture_structure == PICT_BOTTOM_FIELD);

}
