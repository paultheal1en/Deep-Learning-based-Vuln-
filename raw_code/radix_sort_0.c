static void radix_sort(RCCMPEntry *data, int size)

{

    int buckets[RADIX_PASSES][NBUCKETS];

    RCCMPEntry *tmp = av_malloc_array(size, sizeof(*tmp));

    radix_count(data, size, buckets);

    radix_sort_pass(tmp, data, size, buckets[0], 0);

    radix_sort_pass(data, tmp, size, buckets[1], 1);

    if (buckets[2][NBUCKETS - 1] || buckets[3][NBUCKETS - 1]) {

        radix_sort_pass(tmp, data, size, buckets[2], 2);

        radix_sort_pass(data, tmp, size, buckets[3], 3);

    }

    av_free(tmp);

}
