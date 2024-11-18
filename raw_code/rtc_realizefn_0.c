static void rtc_realizefn(DeviceState *dev, Error **errp)

{

    ISADevice *isadev = ISA_DEVICE(dev);

    RTCState *s = MC146818_RTC(dev);

    int base = 0x70;



    s->cmos_data[RTC_REG_A] = 0x26;

    s->cmos_data[RTC_REG_B] = 0x02;

    s->cmos_data[RTC_REG_C] = 0x00;

    s->cmos_data[RTC_REG_D] = 0x80;



    /* This is for historical reasons.  The default base year qdev property

     * was set to 2000 for most machine types before the century byte was

     * implemented.

     *

     * This if statement means that the century byte will be always 0

     * (at least until 2079...) for base_year = 1980, but will be set

     * correctly for base_year = 2000.

     */

    if (s->base_year == 2000) {

        s->base_year = 0;

    }



    rtc_set_date_from_host(isadev);



#ifdef TARGET_I386

    switch (s->lost_tick_policy) {

    case LOST_TICK_POLICY_SLEW:

        s->coalesced_timer =

            timer_new_ns(rtc_clock, rtc_coalesced_timer, s);

        break;

    case LOST_TICK_POLICY_DISCARD:

        break;

    default:

        error_setg(errp, "Invalid lost tick policy.");

        return;

    }

#endif



    s->periodic_timer = timer_new_ns(rtc_clock, rtc_periodic_timer, s);

    s->update_timer = timer_new_ns(rtc_clock, rtc_update_timer, s);

    check_update_timer(s);



    s->clock_reset_notifier.notify = rtc_notify_clock_reset;

    qemu_clock_register_reset_notifier(rtc_clock,

                                       &s->clock_reset_notifier);



    s->suspend_notifier.notify = rtc_notify_suspend;

    qemu_register_suspend_notifier(&s->suspend_notifier);



    memory_region_init_io(&s->io, OBJECT(s), &cmos_ops, s, "rtc", 2);

    isa_register_ioport(isadev, &s->io, base);



    qdev_set_legacy_instance_id(dev, base, 3);

    qemu_register_reset(rtc_reset, s);



    object_property_add(OBJECT(s), "date", "struct tm",

                        rtc_get_date, NULL, NULL, s, NULL);



    object_property_add_alias(qdev_get_machine(), "rtc-time",

                              OBJECT(s), "date", NULL);

}
