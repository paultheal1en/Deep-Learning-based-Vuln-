static BufferPoolEntry *get_pool(AVBufferPool *pool)

{

    BufferPoolEntry *cur = NULL, *last = NULL;



    do {

        FFSWAP(BufferPoolEntry*, cur, last);

        cur = avpriv_atomic_ptr_cas((void * volatile *)&pool->pool, last, NULL);

        if (!cur)

            return NULL;

    } while (cur != last);



    return cur;

}
