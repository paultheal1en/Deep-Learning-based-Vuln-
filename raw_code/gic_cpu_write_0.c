static MemTxResult gic_cpu_write(GICState *s, int cpu, int offset,

                                 uint32_t value, MemTxAttrs attrs)

{

    switch (offset) {

    case 0x00: /* Control */

        gic_set_cpu_control(s, cpu, value, attrs);

        break;

    case 0x04: /* Priority mask */

        s->priority_mask[cpu] = (value & 0xff);

        break;

    case 0x08: /* Binary Point */

        if (s->security_extn && !attrs.secure) {

            s->abpr[cpu] = MAX(value & 0x7, GIC_MIN_ABPR);

        } else {

            s->bpr[cpu] = MAX(value & 0x7, GIC_MIN_BPR);

        }

        break;

    case 0x10: /* End Of Interrupt */

        gic_complete_irq(s, cpu, value & 0x3ff);

        return MEMTX_OK;

    case 0x1c: /* Aliased Binary Point */

        if (!gic_has_groups(s) || (s->security_extn && !attrs.secure)) {

            /* unimplemented, or NS access: RAZ/WI */

            return MEMTX_OK;

        } else {

            s->abpr[cpu] = MAX(value & 0x7, GIC_MIN_ABPR);

        }

        break;

    case 0xd0: case 0xd4: case 0xd8: case 0xdc:

        qemu_log_mask(LOG_UNIMP, "Writing APR not implemented\n");

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "gic_cpu_write: Bad offset %x\n", (int)offset);

        return MEMTX_ERROR;

    }

    gic_update(s);

    return MEMTX_OK;

}
