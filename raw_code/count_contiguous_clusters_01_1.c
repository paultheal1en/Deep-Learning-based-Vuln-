static int count_contiguous_clusters(uint64_t nb_clusters, int cluster_size,

        uint64_t *l2_table, uint64_t mask)

{

    int i;

    uint64_t offset = be64_to_cpu(l2_table[0]) & ~mask;



    if (!offset)

        return 0;



    for (i = 0; i < nb_clusters; i++)

        if (offset + i * cluster_size != (be64_to_cpu(l2_table[i]) & ~mask))

            break;



	return i;

}
