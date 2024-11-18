static uint64_t alloc_cluster_offset(BlockDriverState *bs,

                                     uint64_t offset,

                                     int n_start, int n_end,

                                     int *num)

{

    BDRVQcowState *s = bs->opaque;

    int l2_index, ret;

    uint64_t l2_offset, *l2_table, cluster_offset;

    int nb_available, nb_clusters, i = 0;

    uint64_t start_sect;



    ret = get_cluster_table(bs, offset, &l2_table, &l2_offset, &l2_index);

    if (ret == 0)

        return 0;



    nb_clusters = size_to_clusters(s, n_end << 9);



    if (nb_clusters > s->l2_size - l2_index)

            nb_clusters = s->l2_size - l2_index;



    cluster_offset = be64_to_cpu(l2_table[l2_index]);



    /* We keep all QCOW_OFLAG_COPIED clusters */



    if (cluster_offset & QCOW_OFLAG_COPIED) {

        nb_clusters = count_contiguous_clusters(nb_clusters, s->cluster_size,

                &l2_table[l2_index], 0);



        nb_available = nb_clusters << (s->cluster_bits - 9);

        if (nb_available > n_end)

            nb_available = n_end;



        cluster_offset &= ~QCOW_OFLAG_COPIED;



        goto out;

    }



    /* for the moment, multiple compressed clusters are not managed */



    if (cluster_offset & QCOW_OFLAG_COMPRESSED)

        nb_clusters = 1;



    /* how many available clusters ? */



    while (i < nb_clusters) {

        int j;

        i += count_contiguous_free_clusters(nb_clusters - i,

                &l2_table[l2_index + i]);



        cluster_offset = be64_to_cpu(l2_table[l2_index + i]);



        if ((cluster_offset & QCOW_OFLAG_COPIED) ||

                (cluster_offset & QCOW_OFLAG_COMPRESSED))

            break;



        j = count_contiguous_clusters(nb_clusters - i, s->cluster_size,

                &l2_table[l2_index + i], 0);



        if (j)

            free_any_clusters(bs, cluster_offset, j);



        i += j;



        if(be64_to_cpu(l2_table[l2_index + i]))

            break;

    }

    nb_clusters = i;



    /* allocate a new cluster */



    cluster_offset = alloc_clusters(bs, nb_clusters * s->cluster_size);



    /* we must initialize the cluster content which won't be

       written */



    nb_available = nb_clusters << (s->cluster_bits - 9);

    if (nb_available > n_end)

        nb_available = n_end;



    /* copy content of unmodified sectors */



    start_sect = (offset & ~(s->cluster_size - 1)) >> 9;

    if (n_start) {

        ret = copy_sectors(bs, start_sect, cluster_offset, 0, n_start);

        if (ret < 0)

            return 0;

    }



    if (nb_available & (s->cluster_sectors - 1)) {

        uint64_t end = nb_available & ~(uint64_t)(s->cluster_sectors - 1);

        ret = copy_sectors(bs, start_sect + end,

                           cluster_offset + (end << 9),

                           nb_available - end,

                           s->cluster_sectors);

        if (ret < 0)

            return 0;

    }



    /* update L2 table */



    for (i = 0; i < nb_clusters; i++)

        l2_table[l2_index + i] = cpu_to_be64((cluster_offset +

                                             (i << s->cluster_bits)) |

                                             QCOW_OFLAG_COPIED);



    if (bdrv_pwrite(s->hd,

                    l2_offset + l2_index * sizeof(uint64_t),

                    l2_table + l2_index,

                    nb_clusters * sizeof(uint64_t)) !=

                    nb_clusters * sizeof(uint64_t))

        return 0;



out:

    *num = nb_available - n_start;



    return cluster_offset;

}
