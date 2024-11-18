static int filter_frame(AVFilterLink *inlink, AVFilterBufferRef *in)

{

    HQDN3DContext *hqdn3d = inlink->dst->priv;

    AVFilterLink *outlink = inlink->dst->outputs[0];



    AVFilterBufferRef *out;

    int direct, c;



    if (in->perms & AV_PERM_WRITE) {

        direct = 1;

        out = in;

    } else {

        out = ff_get_video_buffer(outlink, AV_PERM_WRITE, outlink->w, outlink->h);

        if (!out) {

            avfilter_unref_bufferp(&in);

            return AVERROR(ENOMEM);

        }



        avfilter_copy_buffer_ref_props(out, in);

        out->video->w = outlink->w;

        out->video->h = outlink->h;

    }



    for (c = 0; c < 3; c++) {

        denoise(hqdn3d, in->data[c], out->data[c],

                hqdn3d->line, &hqdn3d->frame_prev[c],

                in->video->w >> (!!c * hqdn3d->hsub),

                in->video->h >> (!!c * hqdn3d->vsub),

                in->linesize[c], out->linesize[c],

                hqdn3d->coefs[c?2:0], hqdn3d->coefs[c?3:1]);

    }



    if (!direct)

        avfilter_unref_bufferp(&in);



    return ff_filter_frame(outlink, out);

}
