static uint64_t a9_scu_read(void *opaque, target_phys_addr_t offset,

                            unsigned size)

{

    a9mp_priv_state *s = (a9mp_priv_state *)opaque;

    switch (offset) {

    case 0x00: /* Control */

        return s->scu_control;

    case 0x04: /* Configuration */

        return (((1 << s->num_cpu) - 1) << 4) | (s->num_cpu - 1);

    case 0x08: /* CPU Power Status */

        return s->scu_status;

    case 0x09: /* CPU status.  */

        return s->scu_status >> 8;

    case 0x0a: /* CPU status.  */

        return s->scu_status >> 16;

    case 0x0b: /* CPU status.  */

        return s->scu_status >> 24;

    case 0x0c: /* Invalidate All Registers In Secure State */

        return 0;

    case 0x40: /* Filtering Start Address Register */

    case 0x44: /* Filtering End Address Register */

        /* RAZ/WI, like an implementation with only one AXI master */

        return 0;

    case 0x50: /* SCU Access Control Register */

    case 0x54: /* SCU Non-secure Access Control Register */

        /* unimplemented, fall through */

    default:

        return 0;

    }

}
