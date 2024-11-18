void *g_realloc_n(void *ptr, size_t nmemb, size_t size)

{

    size_t sz;



    __coverity_negative_sink__(nmemb);

    __coverity_negative_sink__(size);

    sz = nmemb * size;

    __coverity_escape__(ptr);

    ptr = __coverity_alloc__(size);

    /*

     * Memory beyond the old size isn't actually initialized.  Can't

     * model that.  See Coverity's realloc() model

     */

    __coverity_writeall__(ptr);

    __coverity_mark_as_afm_allocated__(ptr, AFM_free);

    return ptr;

}
