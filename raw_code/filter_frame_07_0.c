static int filter_frame(AVFilterLink *inlink, AVFrame *in)

{

    AVFilterContext *ctx = inlink->dst;

    HDCDContext *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    AVFrame *out;

    const int16_t *in_data;

    int32_t *out_data;

    int n, c;

    int detect, packets, pe_packets;



    out = ff_get_audio_buffer(outlink, in->nb_samples);

    if (!out) {

        av_frame_free(&in);

        return AVERROR(ENOMEM);

    }

    av_frame_copy_props(out, in);

    out->format = outlink->format;



    in_data  = (int16_t*)in->data[0];

    out_data = (int32_t*)out->data[0];

    for (n = 0; n < in->nb_samples * in->channels; n++) {

        out_data[n] = in_data[n];

    }



    detect = 0;

    packets = 0;

    pe_packets = 0;

    s->det_errors = 0;

    for (c = 0; c < inlink->channels; c++) {

        hdcd_state_t *state = &s->state[c];

        hdcd_process(s, state, out_data + c, in->nb_samples, out->channels);

        if (state->sustain) detect++;

        packets += state->code_counterA + state->code_counterB;

        pe_packets += state->count_peak_extend;

        s->uses_transient_filter |= !!state->count_transient_filter;

        s->max_gain_adjustment = FFMIN(s->max_gain_adjustment, GAINTOFLOAT(state->max_gain));

        s->det_errors += state->code_counterA_almost

            + state->code_counterB_checkfails

            + state->code_counterC_unmatched;

    }

    if (pe_packets) {

        /* if every valid packet has used PE, call it permanent */

        if (packets == pe_packets)

            s->peak_extend = HDCD_PE_PERMANENT;

        else

            s->peak_extend = HDCD_PE_INTERMITTENT;

    } else {

        s->peak_extend = HDCD_PE_NEVER;

    }

    /* HDCD is detected if a valid packet is active in all (both)

     * channels at the same time. */

    if (detect == inlink->channels) s->hdcd_detected = 1;



    s->sample_count += in->nb_samples * in->channels;



    av_frame_free(&in);

    return ff_filter_frame(outlink, out);

}
