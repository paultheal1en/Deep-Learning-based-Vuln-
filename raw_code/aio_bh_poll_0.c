int aio_bh_poll(AioContext *ctx)

{

    QEMUBH *bh, **bhp, *next;

    int ret;

    bool deleted = false;



    qemu_lockcnt_inc(&ctx->list_lock);



    ret = 0;

    for (bh = atomic_rcu_read(&ctx->first_bh); bh; bh = next) {

        next = atomic_rcu_read(&bh->next);

        /* The atomic_xchg is paired with the one in qemu_bh_schedule.  The

         * implicit memory barrier ensures that the callback sees all writes

         * done by the scheduling thread.  It also ensures that the scheduling

         * thread sees the zero before bh->cb has run, and thus will call

         * aio_notify again if necessary.

         */

        if (atomic_xchg(&bh->scheduled, 0)) {

            /* Idle BHs don't count as progress */

            if (!bh->idle) {

                ret = 1;

            }

            bh->idle = 0;

            aio_bh_call(bh);

        }

        if (bh->deleted) {

            deleted = true;

        }

    }



    /* remove deleted bhs */

    if (!deleted) {

        qemu_lockcnt_dec(&ctx->list_lock);

        return ret;

    }



    if (qemu_lockcnt_dec_and_lock(&ctx->list_lock)) {

        bhp = &ctx->first_bh;

        while (*bhp) {

            bh = *bhp;

            if (bh->deleted && !bh->scheduled) {

                *bhp = bh->next;

                g_free(bh);

            } else {

                bhp = &bh->next;

            }

        }

        qemu_lockcnt_unlock(&ctx->list_lock);

    }

    return ret;

}
