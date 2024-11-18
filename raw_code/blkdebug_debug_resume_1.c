static int blkdebug_debug_resume(BlockDriverState *bs, const char *tag)

{

    BDRVBlkdebugState *s = bs->opaque;

    BlkdebugSuspendedReq *r;



    QLIST_FOREACH(r, &s->suspended_reqs, next) {

        if (!strcmp(r->tag, tag)) {

            qemu_coroutine_enter(r->co, NULL);

            return 0;

        }

    }

    return -ENOENT;

}
