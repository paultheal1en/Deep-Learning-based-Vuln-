static void bdrv_aio_co_cancel_em(BlockDriverAIOCB *blockacb)

{

    BlockDriverAIOCBCoroutine *acb =

        container_of(blockacb, BlockDriverAIOCBCoroutine, common);

    bool done = false;



    acb->done = &done;

    while (!done) {

        qemu_aio_wait();

    }

}
