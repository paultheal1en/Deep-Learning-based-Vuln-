void coroutine_fn co_aio_sleep_ns(AioContext *ctx, QEMUClockType type,
                                  int64_t ns)
{
    CoSleepCB sleep_cb = {
        .co = qemu_coroutine_self(),
    };
    sleep_cb.ts = aio_timer_new(ctx, type, SCALE_NS, co_sleep_cb, &sleep_cb);
    timer_mod(sleep_cb.ts, qemu_clock_get_ns(type) + ns);
    qemu_coroutine_yield();
    timer_del(sleep_cb.ts);
    timer_free(sleep_cb.ts);