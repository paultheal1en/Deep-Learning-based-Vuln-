void *qht_do_lookup(struct qht_bucket *head, qht_lookup_func_t func,

                    const void *userp, uint32_t hash)

{

    struct qht_bucket *b = head;

    int i;



    do {

        for (i = 0; i < QHT_BUCKET_ENTRIES; i++) {

            if (b->hashes[i] == hash) {

                /* The pointer is dereferenced before seqlock_read_retry,

                 * so (unlike qht_insert__locked) we need to use

                 * atomic_rcu_read here.

                 */

                void *p = atomic_rcu_read(&b->pointers[i]);



                if (likely(p) && likely(func(p, userp))) {

                    return p;

                }

            }

        }

        b = atomic_rcu_read(&b->next);

    } while (b);



    return NULL;

}
