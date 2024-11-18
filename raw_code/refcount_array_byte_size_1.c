static size_t refcount_array_byte_size(BDRVQcow2State *s, uint64_t entries)

{

    /* This assertion holds because there is no way we can address more than

     * 2^(64 - 9) clusters at once (with cluster size 512 = 2^9, and because

     * offsets have to be representable in bytes); due to every cluster

     * corresponding to one refcount entry, we are well below that limit */

    assert(entries < (UINT64_C(1) << (64 - 9)));



    /* Thanks to the assertion this will not overflow, because

     * s->refcount_order < 7.

     * (note: x << s->refcount_order == x * s->refcount_bits) */

    return DIV_ROUND_UP(entries << s->refcount_order, 8);

}
