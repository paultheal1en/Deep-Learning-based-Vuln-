static void draw_slice(HYuvContext *s, AVFrame *frame, int y)

{

    int h, cy, i;

    int offset[AV_NUM_DATA_POINTERS];



    if (s->avctx->draw_horiz_band == NULL)

        return;



    h  = y - s->last_slice_end;

    y -= h;



    if (s->bitstream_bpp == 12)

        cy = y >> 1;

    else

        cy = y;



    offset[0] = frame->linesize[0] * y;

    offset[1] = frame->linesize[1] * cy;

    offset[2] = frame->linesize[2] * cy;

    for (i = 3; i < AV_NUM_DATA_POINTERS; i++)

        offset[i] = 0;

    emms_c();



    s->avctx->draw_horiz_band(s->avctx, frame, offset, y, 3, h);



    s->last_slice_end = y + h;

}
