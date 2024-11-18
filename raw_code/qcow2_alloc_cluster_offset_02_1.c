int qcow2_alloc_cluster_offset(BlockDriverState *bs, uint64_t offset,

    int *num, uint64_t *host_offset, QCowL2Meta **m)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t start, remaining;

    uint64_t cluster_offset;

    uint64_t cur_bytes;

    int ret;



    trace_qcow2_alloc_clusters_offset(qemu_coroutine_self(), offset, *num);



    assert((offset & ~BDRV_SECTOR_MASK) == 0);



again:

    start = offset;

    remaining = *num << BDRV_SECTOR_BITS;

    cluster_offset = 0;

    *host_offset = 0;

    cur_bytes = 0;

    *m = NULL;



    while (true) {



        if (!*host_offset) {

            *host_offset = start_of_cluster(s, cluster_offset);

        }



        assert(remaining >= cur_bytes);



        start           += cur_bytes;

        remaining       -= cur_bytes;

        cluster_offset  += cur_bytes;



        if (remaining == 0) {

            break;

        }



        cur_bytes = remaining;



        /*

         * Now start gathering as many contiguous clusters as possible:

         *

         * 1. Check for overlaps with in-flight allocations

         *

         *      a) Overlap not in the first cluster -> shorten this request and

         *         let the caller handle the rest in its next loop iteration.

         *

         *      b) Real overlaps of two requests. Yield and restart the search

         *         for contiguous clusters (the situation could have changed

         *         while we were sleeping)

         *

         *      c) TODO: Request starts in the same cluster as the in-flight

         *         allocation ends. Shorten the COW of the in-fight allocation,

         *         set cluster_offset to write to the same cluster and set up

         *         the right synchronisation between the in-flight request and

         *         the new one.

         */

        ret = handle_dependencies(bs, start, &cur_bytes, m);

        if (ret == -EAGAIN) {

            /* Currently handle_dependencies() doesn't yield if we already had

             * an allocation. If it did, we would have to clean up the L2Meta

             * structs before starting over. */

            assert(*m == NULL);

            goto again;

        } else if (ret < 0) {

            return ret;

        } else if (cur_bytes == 0) {

            break;

        } else {

            /* handle_dependencies() may have decreased cur_bytes (shortened

             * the allocations below) so that the next dependency is processed

             * correctly during the next loop iteration. */

        }



        /*

         * 2. Count contiguous COPIED clusters.

         */

        ret = handle_copied(bs, start, &cluster_offset, &cur_bytes, m);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            continue;

        } else if (cur_bytes == 0) {

            break;

        }



        /*

         * 3. If the request still hasn't completed, allocate new clusters,

         *    considering any cluster_offset of steps 1c or 2.

         */

        ret = handle_alloc(bs, start, &cluster_offset, &cur_bytes, m);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            continue;

        } else {

            assert(cur_bytes == 0);

            break;

        }

    }



    *num -= remaining >> BDRV_SECTOR_BITS;

    assert(*num > 0);

    assert(*host_offset != 0);



    return 0;

}
