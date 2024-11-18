int qcow2_update_snapshot_refcount(BlockDriverState *bs,

    int64_t l1_table_offset, int l1_size, int addend)

{

    BDRVQcowState *s = bs->opaque;

    uint64_t *l1_table, *l2_table, l2_offset, offset, l1_size2, l1_allocated;

    int64_t old_offset, old_l2_offset;

    int i, j, l1_modified = 0, nb_csectors, refcount;

    int ret;



    l2_table = NULL;

    l1_table = NULL;

    l1_size2 = l1_size * sizeof(uint64_t);



    /* WARNING: qcow2_snapshot_goto relies on this function not using the

     * l1_table_offset when it is the current s->l1_table_offset! Be careful

     * when changing this! */

    if (l1_table_offset != s->l1_table_offset) {

        l1_table = g_malloc0(align_offset(l1_size2, 512));

        l1_allocated = 1;



        ret = bdrv_pread(bs->file, l1_table_offset, l1_table, l1_size2);

        if (ret < 0) {

            goto fail;

        }



        for(i = 0;i < l1_size; i++)

            be64_to_cpus(&l1_table[i]);

    } else {

        assert(l1_size == s->l1_size);

        l1_table = s->l1_table;

        l1_allocated = 0;

    }



    for(i = 0; i < l1_size; i++) {

        l2_offset = l1_table[i];

        if (l2_offset) {

            old_l2_offset = l2_offset;

            l2_offset &= L1E_OFFSET_MASK;



            ret = qcow2_cache_get(bs, s->l2_table_cache, l2_offset,

                (void**) &l2_table);

            if (ret < 0) {

                goto fail;

            }



            for(j = 0; j < s->l2_size; j++) {

                offset = be64_to_cpu(l2_table[j]);

                if (offset != 0) {

                    old_offset = offset;

                    offset &= ~QCOW_OFLAG_COPIED;

                    if (offset & QCOW_OFLAG_COMPRESSED) {

                        nb_csectors = ((offset >> s->csize_shift) &

                                       s->csize_mask) + 1;

                        if (addend != 0) {

                            int ret;

                            ret = update_refcount(bs,

                                (offset & s->cluster_offset_mask) & ~511,

                                nb_csectors * 512, addend);

                            if (ret < 0) {

                                goto fail;

                            }

                        }

                        /* compressed clusters are never modified */

                        refcount = 2;

                    } else {

                        uint64_t cluster_index = (offset & L2E_OFFSET_MASK) >> s->cluster_bits;

                        if (addend != 0) {

                            refcount = update_cluster_refcount(bs, cluster_index, addend);

                        } else {

                            refcount = get_refcount(bs, cluster_index);

                        }



                        if (refcount < 0) {

                            ret = refcount;

                            goto fail;

                        }

                    }



                    if (refcount == 1) {

                        offset |= QCOW_OFLAG_COPIED;

                    }

                    if (offset != old_offset) {

                        if (addend > 0) {

                            qcow2_cache_set_dependency(bs, s->l2_table_cache,

                                s->refcount_block_cache);

                        }

                        l2_table[j] = cpu_to_be64(offset);

                        qcow2_cache_entry_mark_dirty(s->l2_table_cache, l2_table);

                    }

                }

            }



            ret = qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

            if (ret < 0) {

                goto fail;

            }





            if (addend != 0) {

                refcount = update_cluster_refcount(bs, l2_offset >> s->cluster_bits, addend);

            } else {

                refcount = get_refcount(bs, l2_offset >> s->cluster_bits);

            }

            if (refcount < 0) {

                ret = refcount;

                goto fail;

            } else if (refcount == 1) {

                l2_offset |= QCOW_OFLAG_COPIED;

            }

            if (l2_offset != old_l2_offset) {

                l1_table[i] = l2_offset;

                l1_modified = 1;

            }

        }

    }



    ret = bdrv_flush(bs);

fail:

    if (l2_table) {

        qcow2_cache_put(bs, s->l2_table_cache, (void**) &l2_table);

    }



    /* Update L1 only if it isn't deleted anyway (addend = -1) */

    if (addend >= 0 && l1_modified) {

        for(i = 0; i < l1_size; i++)

            cpu_to_be64s(&l1_table[i]);

        if (bdrv_pwrite_sync(bs->file, l1_table_offset, l1_table,

                        l1_size2) < 0)

            goto fail;

        for(i = 0; i < l1_size; i++)

            be64_to_cpus(&l1_table[i]);

    }

    if (l1_allocated)

        g_free(l1_table);

    return ret;

}
