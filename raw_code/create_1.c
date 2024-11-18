static struct ResampleContext *create(struct ResampleContext *c, int out_rate, int in_rate, int filter_size, int phase_shift, int linear,

        double cutoff, enum AVSampleFormat format, enum SwrFilterType filter_type, double kaiser_beta, double precision, int cheby, int exact_rational){

    soxr_error_t error;



    soxr_datatype_t type =

        format == AV_SAMPLE_FMT_S16P? SOXR_INT16_S :

        format == AV_SAMPLE_FMT_S16 ? SOXR_INT16_I :

        format == AV_SAMPLE_FMT_S32P? SOXR_INT32_S :

        format == AV_SAMPLE_FMT_S32 ? SOXR_INT32_I :

        format == AV_SAMPLE_FMT_FLTP? SOXR_FLOAT32_S :

        format == AV_SAMPLE_FMT_FLT ? SOXR_FLOAT32_I :

        format == AV_SAMPLE_FMT_DBLP? SOXR_FLOAT64_S :

        format == AV_SAMPLE_FMT_DBL ? SOXR_FLOAT64_I : (soxr_datatype_t)-1;



    soxr_io_spec_t io_spec = soxr_io_spec(type, type);



    soxr_quality_spec_t q_spec = soxr_quality_spec((int)((precision-2)/4), (SOXR_HI_PREC_CLOCK|SOXR_ROLLOFF_NONE)*!!cheby);

    q_spec.precision = linear? 0 : precision;

#if !defined SOXR_VERSION /* Deprecated @ March 2013: */

    q_spec.bw_pc = cutoff? FFMAX(FFMIN(cutoff,.995),.8)*100 : q_spec.bw_pc;

#else

    q_spec.passband_end = cutoff? FFMAX(FFMIN(cutoff,.995),.8) : q_spec.passband_end;

#endif



    soxr_delete((soxr_t)c);

    c = (struct ResampleContext *)

        soxr_create(in_rate, out_rate, 0, &error, &io_spec, &q_spec, 0);

    if (!c)

        av_log(NULL, AV_LOG_ERROR, "soxr_create: %s\n", error);

    return c;

}
