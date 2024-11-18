static uint32_t apic_mem_readl(void *opaque, target_phys_addr_t addr)

{

    DeviceState *d;

    APICCommonState *s;

    uint32_t val;

    int index;



    d = cpu_get_current_apic();

    if (!d) {

        return 0;

    }

    s = DO_UPCAST(APICCommonState, busdev.qdev, d);



    index = (addr >> 4) & 0xff;

    switch(index) {

    case 0x02: /* id */

        val = s->id << 24;

        break;

    case 0x03: /* version */

        val = 0x11 | ((APIC_LVT_NB - 1) << 16); /* version 0x11 */

        break;

    case 0x08:

        apic_sync_vapic(s, SYNC_FROM_VAPIC);

        if (apic_report_tpr_access) {

            cpu_report_tpr_access(s->cpu_env, TPR_ACCESS_READ);

        }

        val = s->tpr;

        break;

    case 0x09:

        val = apic_get_arb_pri(s);

        break;

    case 0x0a:

        /* ppr */

        val = apic_get_ppr(s);

        break;

    case 0x0b:

        val = 0;

        break;

    case 0x0d:

        val = s->log_dest << 24;

        break;

    case 0x0e:

        val = s->dest_mode << 28;

        break;

    case 0x0f:

        val = s->spurious_vec;

        break;

    case 0x10 ... 0x17:

        val = s->isr[index & 7];

        break;

    case 0x18 ... 0x1f:

        val = s->tmr[index & 7];

        break;

    case 0x20 ... 0x27:

        val = s->irr[index & 7];

        break;

    case 0x28:

        val = s->esr;

        break;

    case 0x30:

    case 0x31:

        val = s->icr[index & 1];

        break;

    case 0x32 ... 0x37:

        val = s->lvt[index - 0x32];

        break;

    case 0x38:

        val = s->initial_count;

        break;

    case 0x39:

        val = apic_get_current_count(s);

        break;

    case 0x3e:

        val = s->divide_conf;

        break;

    default:

        s->esr |= ESR_ILLEGAL_ADDRESS;

        val = 0;

        break;

    }

    trace_apic_mem_readl(addr, val);

    return val;

}
