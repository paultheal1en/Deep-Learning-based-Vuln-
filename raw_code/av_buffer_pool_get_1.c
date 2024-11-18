AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)

{

    AVBufferRef *ret;

    BufferPoolEntry *buf;



    /* check whether the pool is empty */

    buf = get_pool(pool);

    if (!buf)

        return pool_alloc_buffer(pool);



    /* keep the first entry, return the rest of the list to the pool */

    add_to_pool(buf->next);

    buf->next = NULL;



    ret = av_buffer_create(buf->data, pool->size, pool_release_buffer,

                           buf, 0);

    if (!ret) {

        add_to_pool(buf);

        return NULL;

    }

    avpriv_atomic_int_add_and_fetch(&pool->refcount, 1);



    return ret;

}
