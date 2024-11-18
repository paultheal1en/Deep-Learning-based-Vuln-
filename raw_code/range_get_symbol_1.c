static inline int range_get_symbol(APEContext * ctx,

                                   const uint32_t counts[],

                                   const uint16_t counts_diff[])

{

    int symbol, cf;



    cf = range_decode_culshift(ctx, 16);










    /* figure out the symbol inefficiently; a binary search would be much better */

    for (symbol = 0; counts[symbol + 1] <= cf; symbol++);



    range_decode_update(ctx, counts_diff[symbol], counts[symbol]);



