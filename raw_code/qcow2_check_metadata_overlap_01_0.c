int qcow2_check_metadata_overlap(BlockDriverState *bs, int chk, int64_t offset,

                                 int64_t size)

{

    BDRVQcowState *s = bs->opaque;

    int i, j;



    if (!size) {

        return 0;

    }



    if (chk & QCOW2_OL_MAIN_HEADER) {

        if (offset < s->cluster_size) {

            return QCOW2_OL_MAIN_HEADER;

        }

    }



    /* align range to test to cluster boundaries */

    size = align_offset(offset_into_cluster(s, offset) + size, s->cluster_size);

    offset = start_of_cluster(s, offset);



    if ((chk & QCOW2_OL_ACTIVE_L1) && s->l1_size) {

        if (overlaps_with(s->l1_table_offset, s->l1_size * sizeof(uint64_t))) {

            return QCOW2_OL_ACTIVE_L1;

        }

    }



    if ((chk & QCOW2_OL_REFCOUNT_TABLE) && s->refcount_table_size) {

        if (overlaps_with(s->refcount_table_offset,

            s->refcount_table_size * sizeof(uint64_t))) {

            return QCOW2_OL_REFCOUNT_TABLE;

        }

    }



    if ((chk & QCOW2_OL_SNAPSHOT_TABLE) && s->snapshots_size) {

        if (overlaps_with(s->snapshots_offset, s->snapshots_size)) {

            return QCOW2_OL_SNAPSHOT_TABLE;

        }

    }



    if ((chk & QCOW2_OL_INACTIVE_L1) && s->snapshots) {

        for (i = 0; i < s->nb_snapshots; i++) {

            if (s->snapshots[i].l1_size &&

                overlaps_with(s->snapshots[i].l1_table_offset,

                s->snapshots[i].l1_size * sizeof(uint64_t))) {

                return QCOW2_OL_INACTIVE_L1;

            }

        }

    }



    if ((chk & QCOW2_OL_ACTIVE_L2) && s->l1_table) {

        for (i = 0; i < s->l1_size; i++) {

            if ((s->l1_table[i] & L1E_OFFSET_MASK) &&

                overlaps_with(s->l1_table[i] & L1E_OFFSET_MASK,

                s->cluster_size)) {

                return QCOW2_OL_ACTIVE_L2;

            }

        }

    }



    if ((chk & QCOW2_OL_REFCOUNT_BLOCK) && s->refcount_table) {

        for (i = 0; i < s->refcount_table_size; i++) {

            if ((s->refcount_table[i] & REFT_OFFSET_MASK) &&

                overlaps_with(s->refcount_table[i] & REFT_OFFSET_MASK,

                s->cluster_size)) {

                return QCOW2_OL_REFCOUNT_BLOCK;

            }

        }

    }



    if ((chk & QCOW2_OL_INACTIVE_L2) && s->snapshots) {

        for (i = 0; i < s->nb_snapshots; i++) {

            uint64_t l1_ofs = s->snapshots[i].l1_table_offset;

            uint32_t l1_sz  = s->snapshots[i].l1_size;

            uint64_t *l1 = g_malloc(l1_sz * sizeof(uint64_t));

            int ret;



            ret = bdrv_read(bs->file, l1_ofs / BDRV_SECTOR_SIZE, (uint8_t *)l1,

                            l1_sz * sizeof(uint64_t) / BDRV_SECTOR_SIZE);



            if (ret < 0) {

                g_free(l1);

                return ret;

            }



            for (j = 0; j < l1_sz; j++) {

                if ((l1[j] & L1E_OFFSET_MASK) &&

                    overlaps_with(l1[j] & L1E_OFFSET_MASK, s->cluster_size)) {

                    g_free(l1);

                    return QCOW2_OL_INACTIVE_L2;

                }

            }



            g_free(l1);

        }

    }



    return 0;

}
