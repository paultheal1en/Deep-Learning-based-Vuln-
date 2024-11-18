void *g_try_malloc(size_t n_bytes)

{

    __coverity_negative_sink__(n_bytes);

    return malloc(n_bytes == 0 ? 1 : n_bytes);

}
