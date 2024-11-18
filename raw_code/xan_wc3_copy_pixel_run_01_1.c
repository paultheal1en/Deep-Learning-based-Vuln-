static inline void xan_wc3_copy_pixel_run(XanContext *s, AVFrame *frame,

                                          int x, int y,

                                          int pixel_count, int motion_x,

                                          int motion_y)

{

    int stride;

    int line_inc;

    int curframe_index, prevframe_index;

    int curframe_x, prevframe_x;

    int width = s->avctx->width;

    uint8_t *palette_plane, *prev_palette_plane;



    if (y + motion_y < 0 || y + motion_y >= s->avctx->height ||

        x + motion_x < 0 || x + motion_x >= s->avctx->width)

        return;



    palette_plane = frame->data[0];

    prev_palette_plane = s->last_frame->data[0];

    if (!prev_palette_plane)

        prev_palette_plane = palette_plane;

    stride = frame->linesize[0];

    line_inc = stride - width;

    curframe_index = y * stride + x;

    curframe_x = x;

    prevframe_index = (y + motion_y) * stride + x + motion_x;

    prevframe_x = x + motion_x;



    if (prev_palette_plane == palette_plane && FFABS(curframe_index - prevframe_index) < pixel_count) {

         avpriv_request_sample(s->avctx, "Overlapping copy");

         return ;

    }



    while (pixel_count &&

           curframe_index  < s->frame_size &&

           prevframe_index < s->frame_size) {

        int count = FFMIN3(pixel_count, width - curframe_x,

                           width - prevframe_x);



        memcpy(palette_plane + curframe_index,

               prev_palette_plane + prevframe_index, count);

        pixel_count     -= count;

        curframe_index  += count;

        prevframe_index += count;

        curframe_x      += count;

        prevframe_x     += count;



        if (curframe_x >= width) {

            curframe_index += line_inc;

            curframe_x = 0;

        }



        if (prevframe_x >= width) {

            prevframe_index += line_inc;

            prevframe_x = 0;

        }

    }

}
