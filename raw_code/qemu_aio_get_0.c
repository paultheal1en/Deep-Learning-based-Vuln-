void *qemu_aio_get(const AIOCBInfo *aiocb_info, BlockDriverState *bs,

                   BlockCompletionFunc *cb, void *opaque)

{

    BlockAIOCB *acb;



    acb = g_slice_alloc(aiocb_info->aiocb_size);

    acb->aiocb_info = aiocb_info;

    acb->bs = bs;

    acb->cb = cb;

    acb->opaque = opaque;

    acb->refcnt = 1;

    return acb;

}
