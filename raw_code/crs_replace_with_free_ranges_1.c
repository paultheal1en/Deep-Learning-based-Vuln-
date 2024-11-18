static void crs_replace_with_free_ranges(GPtrArray *ranges,

                                         uint64_t start, uint64_t end)

{

    GPtrArray *free_ranges = g_ptr_array_new_with_free_func(crs_range_free);

    uint64_t free_base = start;

    int i;



    g_ptr_array_sort(ranges, crs_range_compare);

    for (i = 0; i < ranges->len; i++) {

        CrsRangeEntry *used = g_ptr_array_index(ranges, i);



        if (free_base < used->base) {

            crs_range_insert(free_ranges, free_base, used->base - 1);

        }



        free_base = used->limit + 1;

    }



    if (free_base < end) {

        crs_range_insert(free_ranges, free_base, end);

    }



    g_ptr_array_set_size(ranges, 0);

    for (i = 0; i < free_ranges->len; i++) {

        g_ptr_array_add(ranges, g_ptr_array_index(free_ranges, i));

    }



    g_ptr_array_free(free_ranges, false);

}
