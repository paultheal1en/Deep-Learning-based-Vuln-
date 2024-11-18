int av_opt_set_sample_fmt(void *obj, const char *name, enum AVSampleFormat fmt, int search_flags)

{

    return set_format(obj, name, fmt, search_flags, AV_OPT_TYPE_SAMPLE_FMT, "sample", AV_SAMPLE_FMT_NB-1);

}
