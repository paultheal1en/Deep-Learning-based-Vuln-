static uint64_t get_cluster_offset(BlockDriverState *bs,

                                   uint64_t offset, int allocate)

{

    BDRVVmdkState *s = bs->opaque;

    unsigned int l1_index, l2_offset, l2_index;

    int min_index, i, j;

    uint32_t min_count, *l2_table, tmp;

    uint64_t cluster_offset;

    

    l1_index = (offset >> 9) / s->l1_entry_sectors;

    if (l1_index >= s->l1_size)

        return 0;

    l2_offset = s->l1_table[l1_index];

    if (!l2_offset)

        return 0;

    for(i = 0; i < L2_CACHE_SIZE; i++) {

        if (l2_offset == s->l2_cache_offsets[i]) {

            /* increment the hit count */

            if (++s->l2_cache_counts[i] == 0xffffffff) {

                for(j = 0; j < L2_CACHE_SIZE; j++) {

                    s->l2_cache_counts[j] >>= 1;

                }

            }

            l2_table = s->l2_cache + (i * s->l2_size);

            goto found;

        }

    }

    /* not found: load a new entry in the least used one */

    min_index = 0;

    min_count = 0xffffffff;

    for(i = 0; i < L2_CACHE_SIZE; i++) {

        if (s->l2_cache_counts[i] < min_count) {

            min_count = s->l2_cache_counts[i];

            min_index = i;

        }

    }

    l2_table = s->l2_cache + (min_index * s->l2_size);

    if (bdrv_pread(s->hd, (int64_t)l2_offset * 512, l2_table, s->l2_size * sizeof(uint32_t)) != 

                                                                        s->l2_size * sizeof(uint32_t))

        return 0;



    s->l2_cache_offsets[min_index] = l2_offset;

    s->l2_cache_counts[min_index] = 1;

 found:

    l2_index = ((offset >> 9) / s->cluster_sectors) % s->l2_size;

    cluster_offset = le32_to_cpu(l2_table[l2_index]);

    if (!cluster_offset) {

        struct stat file_buf;



        if (!allocate)

            return 0;

        stat(s->hd->filename, &file_buf);

        cluster_offset = file_buf.st_size;

        bdrv_truncate(s->hd, cluster_offset + (s->cluster_sectors << 9));



        cluster_offset >>= 9;

        /* update L2 table */

        tmp = cpu_to_le32(cluster_offset);

        l2_table[l2_index] = tmp;

        if (bdrv_pwrite(s->hd, ((int64_t)l2_offset * 512) + (l2_index * sizeof(tmp)), 

                        &tmp, sizeof(tmp)) != sizeof(tmp))

            return 0;

        /* update backup L2 table */

        if (s->l1_backup_table_offset != 0) {

            l2_offset = s->l1_backup_table[l1_index];

            if (bdrv_pwrite(s->hd, ((int64_t)l2_offset * 512) + (l2_index * sizeof(tmp)), 

                            &tmp, sizeof(tmp)) != sizeof(tmp))

                return 0;

        }



        if (get_whole_cluster(bs, cluster_offset, offset, allocate) == -1)

            return 0;

    }

    cluster_offset <<= 9;

    return cluster_offset;

}
