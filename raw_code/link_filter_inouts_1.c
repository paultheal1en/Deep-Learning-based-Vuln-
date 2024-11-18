static int link_filter_inouts(AVFilterContext *filt_ctx,

                              AVFilterInOut **curr_inputs,

                              AVFilterInOut **open_inputs, AVClass *log_ctx)

{

    int pad = filt_ctx->input_count, ret;



    while (pad--) {

        AVFilterInOut *p = *curr_inputs;

        if (!p) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Not enough inputs specified for the \"%s\" filter.\n",

                   filt_ctx->filter->name);

            return AVERROR(EINVAL);

        }



        *curr_inputs = (*curr_inputs)->next;



        if (p->filter) {

            if ((ret = link_filter(p->filter, p->pad_idx, filt_ctx, pad, log_ctx)) < 0)

                return ret;

            av_free(p->name);

            av_free(p);

        } else {

            p->filter = filt_ctx;

            p->pad_idx = pad;

            insert_inout(open_inputs, p);

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



        currlinkn->filter  = filt_ctx;

        currlinkn->pad_idx = pad;

        insert_inout(curr_inputs, currlinkn);

    }



    return 0;

}