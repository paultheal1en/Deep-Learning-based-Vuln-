BlockReopenQueue *bdrv_reopen_queue(BlockReopenQueue *bs_queue,

                                    BlockDriverState *bs, int flags)

{

    assert(bs != NULL);



    BlockReopenQueueEntry *bs_entry;

    if (bs_queue == NULL) {

        bs_queue = g_new0(BlockReopenQueue, 1);

        QSIMPLEQ_INIT(bs_queue);

    }



    /* bdrv_open() masks this flag out */

    flags &= ~BDRV_O_PROTOCOL;



    if (bs->file) {

        bdrv_reopen_queue(bs_queue, bs->file, bdrv_inherited_flags(flags));

    }



    bs_entry = g_new0(BlockReopenQueueEntry, 1);

    QSIMPLEQ_INSERT_TAIL(bs_queue, bs_entry, entry);



    bs_entry->state.bs = bs;

    bs_entry->state.flags = flags;



    return bs_queue;

}
