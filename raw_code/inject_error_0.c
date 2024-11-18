static int inject_error(BlockDriverState *bs, BlkdebugRule *rule)

{

    BDRVBlkdebugState *s = bs->opaque;

    int error = rule->options.inject.error;

    bool immediately = rule->options.inject.immediately;



    if (rule->options.inject.once) {

        QSIMPLEQ_REMOVE(&s->active_rules, rule, BlkdebugRule, active_next);

        remove_rule(rule);

    }



    if (!immediately) {

        aio_co_schedule(qemu_get_current_aio_context(), qemu_coroutine_self());

        qemu_coroutine_yield();

    }



    return -error;

}
