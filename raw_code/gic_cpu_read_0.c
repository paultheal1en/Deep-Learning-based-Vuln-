static MemTxResult gic_cpu_read(GICState *s, int cpu, int offset,

                                uint64_t *data, MemTxAttrs attrs)

{

    switch (offset) {

    case 0x00: /* Control */

        *data = s->cpu_enabled[cpu];

        break;

    case 0x04: /* Priority mask */

        *data = s->priority_mask[cpu];

        break;

    case 0x08: /* Binary Point */

        if (s->security_extn && !attrs.secure) {

            /* BPR is banked. Non-secure copy stored in ABPR. */

            *data = s->abpr[cpu];

        } else {

            *data = s->bpr[cpu];

        }

        break;

    case 0x0c: /* Acknowledge */

        *data = gic_acknowledge_irq(s, cpu);

        break;

    case 0x14: /* Running Priority */

        *data = s->running_priority[cpu];

        break;

    case 0x18: /* Highest Pending Interrupt */

        *data = s->current_pending[cpu];

        break;

    case 0x1c: /* Aliased Binary Point */

        /* GIC v2, no security: ABPR

         * GIC v1, no security: not implemented (RAZ/WI)

         * With security extensions, secure access: ABPR (alias of NS BPR)

         * With security extensions, nonsecure access: RAZ/WI

         */

        if (!gic_has_groups(s) || (s->security_extn && !attrs.secure)) {

            *data = 0;

        } else {

            *data = s->abpr[cpu];

        }

        break;

    case 0xd0: case 0xd4: case 0xd8: case 0xdc:

        *data = s->apr[(offset - 0xd0) / 4][cpu];

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "gic_cpu_read: Bad offset %x\n", (int)offset);

        return MEMTX_ERROR;

    }

    return MEMTX_OK;

}
