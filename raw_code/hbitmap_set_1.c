void hbitmap_set(HBitmap *hb, uint64_t start, uint64_t count)

{

    /* Compute range in the last layer.  */

    uint64_t last = start + count - 1;



    trace_hbitmap_set(hb, start, count,

                      start >> hb->granularity, last >> hb->granularity);



    start >>= hb->granularity;

    last >>= hb->granularity;

    count = last - start + 1;




    hb->count += count - hb_count_between(hb, start, last);

    hb_set_between(hb, HBITMAP_LEVELS - 1, start, last);

}