bdrv_acct_done(BlockDriverState *bs, BlockAcctCookie *cookie)

{

    assert(cookie->type < BDRV_MAX_IOTYPE);



    bs->stats.nr_bytes[cookie->type] += cookie->bytes;

    bs->stats.nr_ops[cookie->type]++;

    bs->stats.total_time_ns[cookie->type] += get_clock() -

                                             cookie->start_time_ns;

}
