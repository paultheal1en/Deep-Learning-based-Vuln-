static uint32_t cmos_ioport_read(void *opaque, uint32_t addr)

{

    RTCState *s = opaque;

    int ret;

    if ((addr & 1) == 0) {

        return 0xff;

    } else {

        switch(s->cmos_index) {

        case RTC_SECONDS:

        case RTC_MINUTES:

        case RTC_HOURS:

        case RTC_DAY_OF_WEEK:

        case RTC_DAY_OF_MONTH:

        case RTC_MONTH:

        case RTC_YEAR:

            ret = s->cmos_data[s->cmos_index];

            break;

        case RTC_REG_A:

            ret = s->cmos_data[s->cmos_index];

            break;

        case RTC_REG_C:

            ret = s->cmos_data[s->cmos_index];

            qemu_irq_lower(s->irq);

#ifdef TARGET_I386

            if(s->irq_coalesced) {

                apic_reset_irq_delivered();

                qemu_irq_raise(s->irq);

                if (apic_get_irq_delivered())

                    s->irq_coalesced--;

                break;

            }

#endif

            s->cmos_data[RTC_REG_C] = 0x00;

            break;

        default:

            ret = s->cmos_data[s->cmos_index];

            break;

        }

#ifdef DEBUG_CMOS

        printf("cmos: read index=0x%02x val=0x%02x\n",

               s->cmos_index, ret);

#endif

        return ret;

    }

}
