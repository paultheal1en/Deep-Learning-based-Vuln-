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



    /* check that there was at least one common format */

    if(!(ret->format_count = k)) {

        av_free(ret->formats);

        av_free(ret);

        return NULL;

    }



    /* merge and update all the references */

    ret->refs = av_malloc(sizeof(AVFilterFormats**)*(a->refcount+b->refcount));

    for(i = 0; i < a->refcount; i ++) {

        ret->refs[ret->refcount] = a->refs[i];

        *ret->refs[ret->refcount++] = ret;

    }

    for(i = 0; i < b->refcount; i ++) {

        ret->refs[ret->refcount] = b->refs[i];

        *ret->refs[ret->refcount++] = ret;

    }



    av_free(a->refs);

    av_free(a->formats);

    av_free(a);



    av_free(b->refs);

    av_free(b->formats);

    av_free(b);



    return ret;

}
