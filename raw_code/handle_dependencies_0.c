static int handle_dependencies(BlockDriverState *bs, uint64_t guest_offset,

    unsigned int *nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    QCowL2Meta *old_alloc;



    QLIST_FOREACH(old_alloc, &s->cluster_allocs, next_in_flight) {



        uint64_t start = guest_offset >> s->cluster_bits;

        uint64_t end = start + *nb_clusters;

        uint64_t old_start = old_alloc->offset >> s->cluster_bits;

        uint64_t old_end = old_start + old_alloc->nb_clusters;



        if (end < old_start || start > old_end) {

            /* No intersection */

        } else {

            if (start < old_start) {

                /* Stop at the start of a running allocation */

                *nb_clusters = old_start - start;

            } else {

                *nb_clusters = 0;

            }



            if (*nb_clusters == 0) {

                /* Wait for the dependency to complete. We need to recheck

                 * the free/allocated clusters when we continue. */

                qemu_co_mutex_unlock(&s->lock);

                qemu_co_queue_wait(&old_alloc->dependent_requests);

                qemu_co_mutex_lock(&s->lock);

                return -EAGAIN;

            }

        }

    }



    if (!*nb_clusters) {

        abort();

    }



    return 0;

}
