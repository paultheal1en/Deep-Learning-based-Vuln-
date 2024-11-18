static inline PageDesc *page_find(target_ulong index)

{

    PageDesc *p;



    p = l1_map[index >> L2_BITS];

    if (!p)

        return 0;

    return p + (index & (L2_SIZE - 1));

}
