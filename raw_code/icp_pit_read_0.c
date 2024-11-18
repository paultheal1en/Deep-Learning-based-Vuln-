static uint64_t icp_pit_read(void *opaque, target_phys_addr_t offset,

                             unsigned size)

{

    icp_pit_state *s = (icp_pit_state *)opaque;

    int n;



    /* ??? Don't know the PrimeCell ID for this device.  */

    n = offset >> 8;

    if (n > 2) {

        hw_error("%s: Bad timer %d\n", __func__, n);

    }



    return arm_timer_read(s->timer[n], offset & 0xff);

}
