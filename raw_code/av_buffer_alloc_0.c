AVBufferRef *av_buffer_alloc(int size)

{

    AVBufferRef *ret = NULL;

    uint8_t    *data = NULL;



    data = av_malloc(size);

    if (!data)

        return NULL;



    if(CONFIG_MEMORY_POISONING)

        memset(data, 0x2a, size);



    ret = av_buffer_create(data, size, av_buffer_default_free, NULL, 0);

    if (!ret)

        av_freep(&data);



    return ret;

}
