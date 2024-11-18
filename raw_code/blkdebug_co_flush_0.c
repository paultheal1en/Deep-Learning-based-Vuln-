static int blkdebug_co_flush(BlockDriverState *bs)

{

    BDRVBlkdebugState *s = bs->opaque;

    BlkdebugRule *rule = NULL;



    QSIMPLEQ_FOREACH(rule, &s->active_rules, active_next) {

        if (rule->options.inject.offset == -1) {

            break;

        }

    }



    if (rule && rule->options.inject.error) {

        return inject_error(bs, rule);

    }



    return bdrv_co_flush(bs->file->bs);

}
