void *g_malloc(size_t size)

{

    char * p;

    size += 16;

    p = bsd_vmalloc(size);

    *(size_t *)p = size;

    return p + 16;

}
