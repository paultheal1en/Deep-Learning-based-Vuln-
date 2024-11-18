static uint32_t nvic_readl(nvic_state *s, uint32_t offset)

{

    ARMCPU *cpu;

    uint32_t val;

    int irq;



    switch (offset) {

    case 4: /* Interrupt Control Type.  */

        return (s->num_irq / 32) - 1;

    case 0x10: /* SysTick Control and Status.  */

        val = s->systick.control;

        s->systick.control &= ~SYSTICK_COUNTFLAG;

        return val;

    case 0x14: /* SysTick Reload Value.  */

        return s->systick.reload;

    case 0x18: /* SysTick Current Value.  */

        {

            int64_t t;

            if ((s->systick.control & SYSTICK_ENABLE) == 0)

                return 0;

            t = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

            if (t >= s->systick.tick)

                return 0;

            val = ((s->systick.tick - (t + 1)) / systick_scale(s)) + 1;

            /* The interrupt in triggered when the timer reaches zero.

               However the counter is not reloaded until the next clock

               tick.  This is a hack to return zero during the first tick.  */

            if (val > s->systick.reload)

                val = 0;

            return val;

        }

    case 0x1c: /* SysTick Calibration Value.  */

        return 10000;

    case 0xd00: /* CPUID Base.  */

        cpu = ARM_CPU(current_cpu);

        return cpu->env.cp15.c0_cpuid;

    case 0xd04: /* Interrupt Control State.  */

        /* VECTACTIVE */

        val = s->gic.running_irq[0];

        if (val == 1023) {

            val = 0;

        } else if (val >= 32) {

            val -= 16;

        }

        /* RETTOBASE */

        if (s->gic.running_irq[0] == 1023

                || s->gic.last_active[s->gic.running_irq[0]][0] == 1023) {

            val |= (1 << 11);

        }

        /* VECTPENDING */

        if (s->gic.current_pending[0] != 1023)

            val |= (s->gic.current_pending[0] << 12);

        /* ISRPENDING */

        for (irq = 32; irq < s->num_irq; irq++) {

            if (s->gic.irq_state[irq].pending) {

                val |= (1 << 22);

                break;

            }

        }

        /* PENDSTSET */

        if (s->gic.irq_state[ARMV7M_EXCP_SYSTICK].pending)

            val |= (1 << 26);

        /* PENDSVSET */

        if (s->gic.irq_state[ARMV7M_EXCP_PENDSV].pending)

            val |= (1 << 28);

        /* NMIPENDSET */

        if (s->gic.irq_state[ARMV7M_EXCP_NMI].pending)

            val |= (1 << 31);

        return val;

    case 0xd08: /* Vector Table Offset.  */

        cpu = ARM_CPU(current_cpu);

        return cpu->env.v7m.vecbase;

    case 0xd0c: /* Application Interrupt/Reset Control.  */

        return 0xfa05000;

    case 0xd10: /* System Control.  */

        /* TODO: Implement SLEEPONEXIT.  */

        return 0;

    case 0xd14: /* Configuration Control.  */

        /* TODO: Implement Configuration Control bits.  */

        return 0;

    case 0xd24: /* System Handler Status.  */

        val = 0;

        if (s->gic.irq_state[ARMV7M_EXCP_MEM].active) val |= (1 << 0);

        if (s->gic.irq_state[ARMV7M_EXCP_BUS].active) val |= (1 << 1);

        if (s->gic.irq_state[ARMV7M_EXCP_USAGE].active) val |= (1 << 3);

        if (s->gic.irq_state[ARMV7M_EXCP_SVC].active) val |= (1 << 7);

        if (s->gic.irq_state[ARMV7M_EXCP_DEBUG].active) val |= (1 << 8);

        if (s->gic.irq_state[ARMV7M_EXCP_PENDSV].active) val |= (1 << 10);

        if (s->gic.irq_state[ARMV7M_EXCP_SYSTICK].active) val |= (1 << 11);

        if (s->gic.irq_state[ARMV7M_EXCP_USAGE].pending) val |= (1 << 12);

        if (s->gic.irq_state[ARMV7M_EXCP_MEM].pending) val |= (1 << 13);

        if (s->gic.irq_state[ARMV7M_EXCP_BUS].pending) val |= (1 << 14);

        if (s->gic.irq_state[ARMV7M_EXCP_SVC].pending) val |= (1 << 15);

        if (s->gic.irq_state[ARMV7M_EXCP_MEM].enabled) val |= (1 << 16);

        if (s->gic.irq_state[ARMV7M_EXCP_BUS].enabled) val |= (1 << 17);

        if (s->gic.irq_state[ARMV7M_EXCP_USAGE].enabled) val |= (1 << 18);

        return val;

    case 0xd28: /* Configurable Fault Status.  */

        /* TODO: Implement Fault Status.  */

        qemu_log_mask(LOG_UNIMP, "Configurable Fault Status unimplemented\n");

        return 0;

    case 0xd2c: /* Hard Fault Status.  */

    case 0xd30: /* Debug Fault Status.  */

    case 0xd34: /* Mem Manage Address.  */

    case 0xd38: /* Bus Fault Address.  */

    case 0xd3c: /* Aux Fault Status.  */

        /* TODO: Implement fault status registers.  */

        qemu_log_mask(LOG_UNIMP, "Fault status registers unimplemented\n");

        return 0;

    case 0xd40: /* PFR0.  */

        return 0x00000030;

    case 0xd44: /* PRF1.  */

        return 0x00000200;

    case 0xd48: /* DFR0.  */

        return 0x00100000;

    case 0xd4c: /* AFR0.  */

        return 0x00000000;

    case 0xd50: /* MMFR0.  */

        return 0x00000030;

    case 0xd54: /* MMFR1.  */

        return 0x00000000;

    case 0xd58: /* MMFR2.  */

        return 0x00000000;

    case 0xd5c: /* MMFR3.  */

        return 0x00000000;

    case 0xd60: /* ISAR0.  */

        return 0x01141110;

    case 0xd64: /* ISAR1.  */

        return 0x02111000;

    case 0xd68: /* ISAR2.  */

        return 0x21112231;

    case 0xd6c: /* ISAR3.  */

        return 0x01111110;

    case 0xd70: /* ISAR4.  */

        return 0x01310102;

    /* TODO: Implement debug registers.  */

    default:

        qemu_log_mask(LOG_GUEST_ERROR, "NVIC: Bad read offset 0x%x\n", offset);

        return 0;

    }

}
