static int filter_frame(AVFilterLink *inlink, AVFilterBufferRef *cur_buf)

{

    AlphaExtractContext *extract = inlink->dst->priv;

    AVFilterLink *outlink = inlink->dst->outputs[0];

    AVFilterBufferRef *out_buf =

        ff_get_video_buffer(outlink, AV_PERM_WRITE, outlink->w, outlink->h);

    int ret;



    if (!out_buf) {

        ret = AVERROR(ENOMEM);

        goto end;

    }

    avfilter_copy_buffer_ref_props(out_buf, cur_buf);



    if (extract->is_packed_rgb) {

        int x, y;

        uint8_t *pin, *pout;

        for (y = 0; y < out_buf->video->h; y++) {

            pin = cur_buf->data[0] + y * cur_buf->linesize[0] + extract->rgba_map[A];

            pout = out_buf->data[0] + y * out_buf->linesize[0];

            for (x = 0; x < out_buf->video->w; x++) {

                *pout = *pin;

                pout += 1;

                pin += 4;

            }

        }

    } else {

        const int linesize = FFMIN(out_buf->linesize[Y], cur_buf->linesize[A]);

        int y;

        for (y = 0; y < out_buf->video->h; y++) {

            memcpy(out_buf->data[Y] + y * out_buf->linesize[Y],

                   cur_buf->data[A] + y * cur_buf->linesize[A],

                   linesize);

        }

    }



    ret = ff_filter_frame(outlink, out_buf);



end:

    avfilter_unref_buffer(cur_buf);

    return ret;

}
