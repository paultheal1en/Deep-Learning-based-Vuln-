static void apply_loop_filter(Vp3DecodeContext *s, int plane, int ystart, int yend)

{

    int x, y;

    int *bounding_values= s->bounding_values_array+127;



    int width           = s->fragment_width[!!plane];

    int height          = s->fragment_height[!!plane];

    int fragment        = s->fragment_start        [plane] + ystart * width;

    int stride          = s->current_frame.linesize[plane];

    uint8_t *plane_data = s->current_frame.data    [plane];

    if (!s->flipped_image) stride = -stride;

    plane_data += s->data_offset[plane] + 8*ystart*stride;



    for (y = ystart; y < yend; y++) {



        for (x = 0; x < width; x++) {

            /* This code basically just deblocks on the edges of coded blocks.

             * However, it has to be much more complicated because of the

             * braindamaged deblock ordering used in VP3/Theora. Order matters

             * because some pixels get filtered twice. */

            if( s->all_fragments[fragment].coding_method != MODE_COPY )

            {

                /* do not perform left edge filter for left columns frags */

                if (x > 0) {

                    s->dsp.vp3_h_loop_filter(

                        plane_data + 8*x,

                        stride, bounding_values);

                }



                /* do not perform top edge filter for top row fragments */

                if (y > 0) {

                    s->dsp.vp3_v_loop_filter(

                        plane_data + 8*x,

                        stride, bounding_values);

                }



                /* do not perform right edge filter for right column

                 * fragments or if right fragment neighbor is also coded

                 * in this frame (it will be filtered in next iteration) */

                if ((x < width - 1) &&

                    (s->all_fragments[fragment + 1].coding_method == MODE_COPY)) {

                    s->dsp.vp3_h_loop_filter(

                        plane_data + 8*x + 8,

                        stride, bounding_values);

                }



                /* do not perform bottom edge filter for bottom row

                 * fragments or if bottom fragment neighbor is also coded

                 * in this frame (it will be filtered in the next row) */

                if ((y < height - 1) &&

                    (s->all_fragments[fragment + width].coding_method == MODE_COPY)) {

                    s->dsp.vp3_v_loop_filter(

                        plane_data + 8*x + 8*stride,

                        stride, bounding_values);

                }

            }



            fragment++;

        }

        plane_data += 8*stride;

    }

}
