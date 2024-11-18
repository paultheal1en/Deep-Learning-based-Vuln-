void *av_realloc(void *ptr, size_t size)

{

#if CONFIG_MEMALIGN_HACK

    int diff;

#endif



    /* let's disallow possible ambiguous cases */

    if (size > (MAX_MALLOC_SIZE-16))

        return NULL;



#if CONFIG_MEMALIGN_HACK

    //FIXME this isn't aligned correctly, though it probably isn't needed

    if(!ptr) return av_malloc(size);

    diff= ((char*)ptr)[-1];

    return (char*)realloc((char*)ptr - diff, size + diff) + diff;

#else

    return realloc(ptr, size + !size);

#endif

}
