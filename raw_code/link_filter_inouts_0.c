static int link_filter_inouts(AVFilterContext *filt_ctx,

                              AVFilterInOut **curr_inputs,

                              AVFilterInOut **open_inputs, void *log_ctx)

{

    int pad, ret;



    for (pad = 0; pad < filt_ctx->input_count; pad++) {

        AVFilterInOut *p = *curr_inputs;



        if (p)

            *curr_inputs = (*curr_inputs)->next;

        else if (!(p = av_mallocz(sizeof(*p))))

            return AVERROR(ENOMEM);



        if (p->filter_ctx) {

            if ((ret = link_filter(p->filter_ctx, p->pad_idx, filt_ctx, pad, log_ctx)) < 0)

                return ret;

            av_free(p->name);

            av_free(p);

        } else {

            p->filter_ctx = filt_ctx;

            p->pad_idx = pad;

            append_inout(open_inputs, &p);

        }

    }



    if (*curr_inputs) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Too many inputs specified for the \"%s\" filter.\n",

               filt_ctx->filter->name);

        return AVERROR(EINVAL);

    }



    pad = filt_ctx->output_count;

    while (pad--) {

        AVFilterInOut *currlinkn = av_mallocz(sizeof(AVFilterInOut));

        if (!currlinkn)

            return AVERROR(ENOMEM);

        currlinkn->filter_ctx  = filt_ctx;

        currlinkn->pad_idx = pad;

        insert_inout(curr_inputs, currlinkn);

    }



    return 0;

}
