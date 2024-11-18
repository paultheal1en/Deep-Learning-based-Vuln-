AVFilterFormats *avfilter_merge_formats(AVFilterFormats *a, AVFilterFormats *b)
{
    AVFilterFormats *ret;
    unsigned i, j, k = 0;
    ret = av_mallocz(sizeof(AVFilterFormats));
    /* merge list of formats */
    ret->formats = av_malloc(sizeof(*ret->formats) * FFMIN(a->format_count,
                                                           b->format_count));
    for(i = 0; i < a->format_count; i ++)
        for(j = 0; j < b->format_count; j ++)
            if(a->formats[i] == b->formats[j])
                ret->formats[k++] = a->formats[i];
    ret->format_count = k;
    /* check that there was at least one common format */
    if(!ret->format_count) {
        av_free(ret->formats);
        av_free(ret);
        return NULL;
    }
    ret->refs = av_malloc(sizeof(AVFilterFormats**)*(a->refcount+b->refcount));
    merge_ref(ret, a);
    merge_ref(ret, b);
    return ret;
}