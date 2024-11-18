static void icp_pit_write(void *opaque, hwaddr offset,

                          uint64_t value, unsigned size)

{

    icp_pit_state *s = (icp_pit_state *)opaque;

    int n;



    n = offset >> 8;

    if (n > 2) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: Bad timer %d\n", __func__, n);


    }



    arm_timer_write(s->timer[n], offset & 0xff, value);

}