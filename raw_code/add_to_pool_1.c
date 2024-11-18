static void add_to_pool(BufferPoolEntry *buf)

{

    AVBufferPool *pool;

    BufferPoolEntry *cur, *end = buf;



    if (!buf)

        return;

    pool = buf->pool;



    while (end->next)

        end = end->next;



    while ((cur = avpriv_atomic_ptr_cas((void * volatile *)&pool->pool, NULL, buf))) {

        /* pool is not empty, retrieve it and append it to our list */

        cur = get_pool(pool);

        end->next = cur;

        while (end->next)

            end = end->next;

    }

}
