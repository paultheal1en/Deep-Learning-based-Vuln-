static int check_refblocks(BlockDriverState *bs, BdrvCheckResult *res,

                           BdrvCheckMode fix, uint16_t **refcount_table,

                           int64_t *nb_clusters)

{

    BDRVQcowState *s = bs->opaque;

    int64_t i;



    for(i = 0; i < s->refcount_table_size; i++) {

        uint64_t offset, cluster;

        offset = s->refcount_table[i];

        cluster = offset >> s->cluster_bits;



        /* Refcount blocks are cluster aligned */

        if (offset_into_cluster(s, offset)) {

            fprintf(stderr, "ERROR refcount block %" PRId64 " is not "

                "cluster aligned; refcount table entry corrupted\n", i);

            res->corruptions++;

            continue;

        }



        if (cluster >= *nb_clusters) {

            fprintf(stderr, "ERROR refcount block %" PRId64

                    " is outside image\n", i);

            res->corruptions++;

            continue;

        }



        if (offset != 0) {

            inc_refcounts(bs, res, *refcount_table, *nb_clusters,

                offset, s->cluster_size);

            if ((*refcount_table)[cluster] != 1) {

                fprintf(stderr, "%s refcount block %" PRId64

                    " refcount=%d\n",

                    fix & BDRV_FIX_ERRORS ? "Repairing" :

                                            "ERROR",

                    i, (*refcount_table)[cluster]);



                if (fix & BDRV_FIX_ERRORS) {

                    int64_t new_offset;



                    new_offset = realloc_refcount_block(bs, i, offset);

                    if (new_offset < 0) {

                        res->corruptions++;

                        continue;

                    }



                    /* update refcounts */

                    if ((new_offset >> s->cluster_bits) >= *nb_clusters) {

                        /* increase refcount_table size if necessary */

                        int old_nb_clusters = *nb_clusters;

                        *nb_clusters = (new_offset >> s->cluster_bits) + 1;

                        *refcount_table = g_renew(uint16_t, *refcount_table,

                                                  *nb_clusters);

                        memset(&(*refcount_table)[old_nb_clusters], 0,

                               (*nb_clusters - old_nb_clusters) *

                               sizeof(uint16_t));

                    }

                    (*refcount_table)[cluster]--;

                    inc_refcounts(bs, res, *refcount_table, *nb_clusters,

                            new_offset, s->cluster_size);



                    res->corruptions_fixed++;

                } else {

                    res->corruptions++;

                }

            }

        }

    }



    return 0;

}
