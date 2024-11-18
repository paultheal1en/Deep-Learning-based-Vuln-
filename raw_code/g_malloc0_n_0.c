void *g_malloc0_n(size_t nmemb, size_t size)

{

    size_t sz;

    void *ptr;



    __coverity_negative_sink__(nmemb);

    __coverity_negative_sink__(size);

    sz = nmemb * size;

    ptr = __coverity_alloc__(size);

    __coverity_writeall0__(ptr);

    __coverity_mark_as_afm_allocated__(ptr, AFM_free);

    return ptr;

}
