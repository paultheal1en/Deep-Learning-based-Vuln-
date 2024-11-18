static int draw_slice(AVFilterLink *inlink, int y0, int h, int slice_dir)

{

    AlphaExtractContext *extract = inlink->dst->priv;

    AVFilterBufferRef *cur_buf = inlink->cur_buf;

    AVFilterBufferRef *out_buf = inlink->dst->outputs[0]->out_buf;



    if (extract->is_packed_rgb) {

        int x, y;

        uint8_t *pin, *pout;

        for (y = y0; y < (y0 + h); y++) {

            pin = cur_buf->data[0] + y * cur_buf->linesize[0] + extract->rgba_map[A];

            pout = out_buf->data[0] + y * out_buf->linesize[0];

            for (x = 0; x < out_buf->video->w; x++) {

                *pout = *pin;

                pout += 1;

                pin += 4;

            }

        }

    } else if (cur_buf->linesize[A] == out_buf->linesize[Y]) {

        const int linesize = cur_buf->linesize[A];

        memcpy(out_buf->data[Y] + y0 * linesize,

               cur_buf->data[A] + y0 * linesize,

               linesize * h);

    } else {

        const int linesize = FFMIN(out_buf->linesize[Y], cur_buf->linesize[A]);

        int y;

        for (y = y0; y < (y0 + h); y++) {

            memcpy(out_buf->data[Y] + y * out_buf->linesize[Y],

                   cur_buf->data[A] + y * cur_buf->linesize[A],

                   linesize);

        }

    }

    return ff_draw_slice(inlink->dst->outputs[0], y0, h, slice_dir);

}
