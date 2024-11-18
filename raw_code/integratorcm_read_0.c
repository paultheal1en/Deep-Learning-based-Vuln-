static uint64_t integratorcm_read(void *opaque, target_phys_addr_t offset,

                                  unsigned size)

{

    integratorcm_state *s = (integratorcm_state *)opaque;

    if (offset >= 0x100 && offset < 0x200) {

        /* CM_SPD */

        if (offset >= 0x180)

            return 0;

        return integrator_spd[offset >> 2];

    }

    switch (offset >> 2) {

    case 0: /* CM_ID */

        return 0x411a3001;

    case 1: /* CM_PROC */

        return 0;

    case 2: /* CM_OSC */

        return s->cm_osc;

    case 3: /* CM_CTRL */

        return s->cm_ctrl;

    case 4: /* CM_STAT */

        return 0x00100000;

    case 5: /* CM_LOCK */

        if (s->cm_lock == 0xa05f) {

            return 0x1a05f;

        } else {

            return s->cm_lock;

        }

    case 6: /* CM_LMBUSCNT */

        /* ??? High frequency timer.  */

        hw_error("integratorcm_read: CM_LMBUSCNT");

    case 7: /* CM_AUXOSC */

        return s->cm_auxosc;

    case 8: /* CM_SDRAM */

        return s->cm_sdram;

    case 9: /* CM_INIT */

        return s->cm_init;

    case 10: /* CM_REFCT */

        /* ??? High frequency timer.  */

        hw_error("integratorcm_read: CM_REFCT");

    case 12: /* CM_FLAGS */

        return s->cm_flags;

    case 14: /* CM_NVFLAGS */

        return s->cm_nvflags;

    case 16: /* CM_IRQ_STAT */

        return s->int_level & s->irq_enabled;

    case 17: /* CM_IRQ_RSTAT */

        return s->int_level;

    case 18: /* CM_IRQ_ENSET */

        return s->irq_enabled;

    case 20: /* CM_SOFT_INTSET */

        return s->int_level & 1;

    case 24: /* CM_FIQ_STAT */

        return s->int_level & s->fiq_enabled;

    case 25: /* CM_FIQ_RSTAT */

        return s->int_level;

    case 26: /* CM_FIQ_ENSET */

        return s->fiq_enabled;

    case 32: /* CM_VOLTAGE_CTL0 */

    case 33: /* CM_VOLTAGE_CTL1 */

    case 34: /* CM_VOLTAGE_CTL2 */

    case 35: /* CM_VOLTAGE_CTL3 */

        /* ??? Voltage control unimplemented.  */

        return 0;

    default:

        hw_error("integratorcm_read: Unimplemented offset 0x%x\n",

                 (int)offset);

        return 0;

    }

}
