static void integratorcm_write(void *opaque, target_phys_addr_t offset,

                               uint64_t value, unsigned size)

{

    integratorcm_state *s = (integratorcm_state *)opaque;

    switch (offset >> 2) {

    case 2: /* CM_OSC */

        if (s->cm_lock == 0xa05f)

            s->cm_osc = value;

        break;

    case 3: /* CM_CTRL */

        integratorcm_set_ctrl(s, value);

        break;

    case 5: /* CM_LOCK */

        s->cm_lock = value & 0xffff;

        break;

    case 7: /* CM_AUXOSC */

        if (s->cm_lock == 0xa05f)

            s->cm_auxosc = value;

        break;

    case 8: /* CM_SDRAM */

        s->cm_sdram = value;

        break;

    case 9: /* CM_INIT */

        /* ??? This can change the memory bus frequency.  */

        s->cm_init = value;

        break;

    case 12: /* CM_FLAGSS */

        s->cm_flags |= value;

        break;

    case 13: /* CM_FLAGSC */

        s->cm_flags &= ~value;

        break;

    case 14: /* CM_NVFLAGSS */

        s->cm_nvflags |= value;

        break;

    case 15: /* CM_NVFLAGSS */

        s->cm_nvflags &= ~value;

        break;

    case 18: /* CM_IRQ_ENSET */

        s->irq_enabled |= value;

        integratorcm_update(s);

        break;

    case 19: /* CM_IRQ_ENCLR */

        s->irq_enabled &= ~value;

        integratorcm_update(s);

        break;

    case 20: /* CM_SOFT_INTSET */

        s->int_level |= (value & 1);

        integratorcm_update(s);

        break;

    case 21: /* CM_SOFT_INTCLR */

        s->int_level &= ~(value & 1);

        integratorcm_update(s);

        break;

    case 26: /* CM_FIQ_ENSET */

        s->fiq_enabled |= value;

        integratorcm_update(s);

        break;

    case 27: /* CM_FIQ_ENCLR */

        s->fiq_enabled &= ~value;

        integratorcm_update(s);

        break;

    case 32: /* CM_VOLTAGE_CTL0 */

    case 33: /* CM_VOLTAGE_CTL1 */

    case 34: /* CM_VOLTAGE_CTL2 */

    case 35: /* CM_VOLTAGE_CTL3 */

        /* ??? Voltage control unimplemented.  */

        break;

    default:

        hw_error("integratorcm_write: Unimplemented offset 0x%x\n",

                 (int)offset);

        break;

    }

}
