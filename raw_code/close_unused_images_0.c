static void close_unused_images(BlockDriverState *top, BlockDriverState *base,

                                const char *base_id)

{

    BlockDriverState *intermediate;

    intermediate = top->backing_hd;



    while (intermediate) {

        BlockDriverState *unused;



        /* reached base */

        if (intermediate == base) {

            break;

        }



        unused = intermediate;

        intermediate = intermediate->backing_hd;

        unused->backing_hd = NULL;

        bdrv_delete(unused);

    }

    top->backing_hd = base;

}
