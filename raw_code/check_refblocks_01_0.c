static int check_refblocks(BlockDriverState *bs, BdrvCheckResult *res,

                           BdrvCheckMode fix, bool *rebuild,

                           void **refcount_table, int64_t *nb_clusters)

{

    BDRVQcow2State *s = bs->opaque;

    int64_t i, size;

    int ret;



    for(i = 0; i < s->refcount_table_size; i++) {

        uint64_t offset, cluster;

        offset = s->refcount_table[i];

        cluster = offset >> s->cluster_bits;



        /* Refcount blocks are cluster aligned */

        if (offset_into_cluster(s, offset)) {

            fprintf(stderr, "ERROR refcount block %" PRId64 " is not "

                "cluster aligned; refcount table entry corrupted\n", i);

            res->corruptions++;

            *rebuild = true;

            continue;

        }



        if (cluster >= *nb_clusters) {

            fprintf(stderr, "%s refcount block %" PRId64 " is outside image\n",

                    fix & BDRV_FIX_ERRORS ? "Repairing" : "ERROR", i);



            if (fix & BDRV_FIX_ERRORS) {

                int64_t new_nb_clusters;

                Error *local_err = NULL;



                if (offset > INT64_MAX - s->cluster_size) {

                    ret = -EINVAL;

                    goto resize_fail;

                }



                ret = bdrv_truncate(bs->file, offset + s->cluster_size,

                                    &local_err);

                if (ret < 0) {

                    error_report_err(local_err);

                    goto resize_fail;

                }

                size = bdrv_getlength(bs->file->bs);

                if (size < 0) {

                    ret = size;

                    goto resize_fail;

                }



                new_nb_clusters = size_to_clusters(s, size);

                assert(new_nb_clusters >= *nb_clusters);



                ret = realloc_refcount_array(s, refcount_table,

                                             nb_clusters, new_nb_clusters);

                if (ret < 0) {

                    res->check_errors++;

                    return ret;

                }



                if (cluster >= *nb_clusters) {

                    ret = -EINVAL;

                    goto resize_fail;

                }



                res->corruptions_fixed++;

                ret = inc_refcounts(bs, res, refcount_table, nb_clusters,

                                    offset, s->cluster_size);

                if (ret < 0) {

                    return ret;

                }

                /* No need to check whether the refcount is now greater than 1:

                 * This area was just allocated and zeroed, so it can only be

                 * exactly 1 after inc_refcounts() */

                continue;



resize_fail:

                res->corruptions++;

                *rebuild = true;

                fprintf(stderr, "ERROR could not resize image: %s\n",

                        strerror(-ret));

            } else {

                res->corruptions++;

            }

            continue;

        }



        if (offset != 0) {

            ret = inc_refcounts(bs, res, refcount_table, nb_clusters,

                                offset, s->cluster_size);

            if (ret < 0) {

                return ret;

            }

            if (s->get_refcount(*refcount_table, cluster) != 1) {

                fprintf(stderr, "ERROR refcount block %" PRId64

                        " refcount=%" PRIu64 "\n", i,

                        s->get_refcount(*refcount_table, cluster));

                res->corruptions++;

                *rebuild = true;

            }

        }

    }



    return 0;

}
