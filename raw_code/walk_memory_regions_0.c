int walk_memory_regions(void *priv, walk_memory_regions_fn fn)

{

    struct walk_memory_regions_data data;

    uintptr_t i;



    data.fn = fn;

    data.priv = priv;

    data.start = -1ul;

    data.prot = 0;



    for (i = 0; i < V_L1_SIZE; i++) {

        int rc = walk_memory_regions_1(&data, (abi_ulong)i << V_L1_SHIFT,

                                       V_L1_SHIFT / L2_BITS - 1, l1_map + i);



        if (rc != 0) {

            return rc;

        }

    }



    return walk_memory_regions_end(&data, 0, 0);

}
