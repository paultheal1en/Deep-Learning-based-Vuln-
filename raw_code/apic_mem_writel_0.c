static void apic_mem_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    DeviceState *d;

    APICCommonState *s;

    int index = (addr >> 4) & 0xff;

    if (addr > 0xfff || !index) {

        /* MSI and MMIO APIC are at the same memory location,

         * but actually not on the global bus: MSI is on PCI bus

         * APIC is connected directly to the CPU.

         * Mapping them on the global bus happens to work because

         * MSI registers are reserved in APIC MMIO and vice versa. */

        apic_send_msi(addr, val);

        return;

    }



    d = cpu_get_current_apic();

    if (!d) {

        return;

    }

    s = DO_UPCAST(APICCommonState, busdev.qdev, d);



    trace_apic_mem_writel(addr, val);



    switch(index) {

    case 0x02:

        s->id = (val >> 24);

        break;

    case 0x03:

        break;

    case 0x08:

        if (apic_report_tpr_access) {

            cpu_report_tpr_access(s->cpu_env, TPR_ACCESS_WRITE);

        }

        s->tpr = val;

        apic_sync_vapic(s, SYNC_TO_VAPIC);

        apic_update_irq(s);

        break;

    case 0x09:

    case 0x0a:

        break;

    case 0x0b: /* EOI */

        apic_eoi(s);

        break;

    case 0x0d:

        s->log_dest = val >> 24;

        break;

    case 0x0e:

        s->dest_mode = val >> 28;

        break;

    case 0x0f:

        s->spurious_vec = val & 0x1ff;

        apic_update_irq(s);

        break;

    case 0x10 ... 0x17:

    case 0x18 ... 0x1f:

    case 0x20 ... 0x27:

    case 0x28:

        break;

    case 0x30:

        s->icr[0] = val;

        apic_deliver(d, (s->icr[1] >> 24) & 0xff, (s->icr[0] >> 11) & 1,

                     (s->icr[0] >> 8) & 7, (s->icr[0] & 0xff),

                     (s->icr[0] >> 15) & 1);

        break;

    case 0x31:

        s->icr[1] = val;

        break;

    case 0x32 ... 0x37:

        {

            int n = index - 0x32;

            s->lvt[n] = val;

            if (n == APIC_LVT_TIMER) {

                apic_timer_update(s, qemu_get_clock_ns(vm_clock));

            } else if (n == APIC_LVT_LINT0 && apic_check_pic(s)) {

                apic_update_irq(s);

            }

        }

        break;

    case 0x38:

        s->initial_count = val;

        s->initial_count_load_time = qemu_get_clock_ns(vm_clock);

        apic_timer_update(s, s->initial_count_load_time);

        break;

    case 0x39:

        break;

    case 0x3e:

        {

            int v;

            s->divide_conf = val & 0xb;

            v = (s->divide_conf & 3) | ((s->divide_conf >> 1) & 4);

            s->count_shift = (v + 1) & 7;

        }

        break;

    default:

        s->esr |= ESR_ILLEGAL_ADDRESS;

        break;

    }

}
