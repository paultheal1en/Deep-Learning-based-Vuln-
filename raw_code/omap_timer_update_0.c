static inline void omap_timer_update(struct omap_mpu_timer_s *timer)

{

    int64_t expires;



    if (timer->enable && timer->st && timer->rate) {

        timer->val = timer->reset_val;	/* Should skip this on clk enable */

        expires = timer->time + muldiv64(timer->val << (timer->ptv + 1),

                        ticks_per_sec, timer->rate);

        qemu_mod_timer(timer->timer, expires);

    } else

        qemu_del_timer(timer->timer);

}
