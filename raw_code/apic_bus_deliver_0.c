static void apic_bus_deliver(const uint32_t *deliver_bitmask,

                             uint8_t delivery_mode,

                             uint8_t vector_num, uint8_t polarity,

                             uint8_t trigger_mode)

{

    APICState *apic_iter;



    switch (delivery_mode) {

        case APIC_DM_LOWPRI:

            /* XXX: search for focus processor, arbitration */

            {

                int i, d;

                d = -1;

                for(i = 0; i < MAX_APIC_WORDS; i++) {

                    if (deliver_bitmask[i]) {

                        d = i * 32 + ffs_bit(deliver_bitmask[i]);

                        break;

                    }

                }

                if (d >= 0) {

                    apic_iter = local_apics[d];

                    if (apic_iter) {

                        apic_set_irq(apic_iter, vector_num, trigger_mode);

                    }

                }

            }

            return;



        case APIC_DM_FIXED:

            break;



        case APIC_DM_SMI:

            foreach_apic(apic_iter, deliver_bitmask,

                cpu_interrupt(apic_iter->cpu_env, CPU_INTERRUPT_SMI) );

            return;



        case APIC_DM_NMI:

            foreach_apic(apic_iter, deliver_bitmask,

                cpu_interrupt(apic_iter->cpu_env, CPU_INTERRUPT_NMI) );

            return;



        case APIC_DM_INIT:

            /* normal INIT IPI sent to processors */

            foreach_apic(apic_iter, deliver_bitmask,

                         apic_init_ipi(apic_iter) );

            return;



        case APIC_DM_EXTINT:

            /* handled in I/O APIC code */

            break;



        default:

            return;

    }



    foreach_apic(apic_iter, deliver_bitmask,

                 apic_set_irq(apic_iter, vector_num, trigger_mode) );

}
