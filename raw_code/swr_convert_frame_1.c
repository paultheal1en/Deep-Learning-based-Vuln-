int swr_convert_frame(SwrContext *s,

                      AVFrame *out, const AVFrame *in)

{

    int ret, setup = 0;



    if (!swr_is_initialized(s)) {

        if ((ret = swr_config_frame(s, out, in)) < 0)

            return ret;

        if ((ret = swr_init(s)) < 0)

            return ret;

        setup = 1;

    } else {

        // return as is or reconfigure for input changes?

        if ((ret = config_changed(s, out, in)))

            return ret;

    }



    if (out) {

        if (!out->linesize[0]) {

            out->nb_samples =   swr_get_delay(s, s->out_sample_rate)

                              + in->nb_samples*(int64_t)s->out_sample_rate / s->in_sample_rate

                              + 3;

            if ((ret = av_frame_get_buffer(out, 0)) < 0) {

                if (setup)

                    swr_close(s);

                return ret;

            }

        } else {

            if (!out->nb_samples)

                out->nb_samples = available_samples(out);

        }

    }



    return convert_frame(s, out, in);

}
