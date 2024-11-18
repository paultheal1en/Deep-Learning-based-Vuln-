static void end_frame(AVFilterLink *inlink)

{

    TransContext *trans = inlink->dst->priv;

    AVFilterBufferRef *inpic  = inlink->cur_buf;

    AVFilterBufferRef *outpic = inlink->dst->outputs[0]->out_buf;

    AVFilterLink *outlink = inlink->dst->outputs[0];

    int plane;



    for (plane = 0; outpic->data[plane]; plane++) {

        int hsub = plane == 1 || plane == 2 ? trans->hsub : 0;

        int vsub = plane == 1 || plane == 2 ? trans->vsub : 0;

        int pixstep = trans->pixsteps[plane];

        int inh  = inpic->video->h>>vsub;

        int outw = outpic->video->w>>hsub;

        int outh = outpic->video->h>>vsub;

        uint8_t *out, *in;

        int outlinesize, inlinesize;

        int x, y;



        out = outpic->data[plane]; outlinesize = outpic->linesize[plane];

        in  = inpic ->data[plane]; inlinesize  = inpic ->linesize[plane];



        if (trans->dir&1) {

            in +=  inpic->linesize[plane] * (inh-1);

            inlinesize *= -1;

        }



        if (trans->dir&2) {

            out += outpic->linesize[plane] * (outh-1);

            outlinesize *= -1;

        }



        for (y = 0; y < outh; y++) {

            switch (pixstep) {

            case 1:

                for (x = 0; x < outw; x++)

                    out[x] = in[x*inlinesize + y];

                break;

            case 2:

                for (x = 0; x < outw; x++)

                    *((uint16_t *)(out + 2*x)) = *((uint16_t *)(in + x*inlinesize + y*2));

                break;

            case 3:

                for (x = 0; x < outw; x++) {

                    int32_t v = AV_RB24(in + x*inlinesize + y*3);

                    AV_WB24(out + 3*x, v);

                }

                break;

            case 4:

                for (x = 0; x < outw; x++)

                    *((uint32_t *)(out + 4*x)) = *((uint32_t *)(in + x*inlinesize + y*4));

                break;

            }

            out += outlinesize;

        }

    }



    avfilter_unref_buffer(inpic);

    ff_draw_slice(outlink, 0, outpic->video->h, 1);

    ff_end_frame(outlink);

    avfilter_unref_buffer(outpic);

}
