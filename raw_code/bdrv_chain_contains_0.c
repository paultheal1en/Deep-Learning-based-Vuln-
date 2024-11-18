bool bdrv_chain_contains(BlockDriverState *top, BlockDriverState *base)

{

    while (top && top != base) {

        top = top->backing_hd;

    }



    return top != NULL;

}
