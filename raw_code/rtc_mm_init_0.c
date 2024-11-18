RTCState *rtc_mm_init(target_phys_addr_t base, int it_shift, qemu_irq irq,

                      int base_year)

{

    RTCState *s;

    int io_memory;



    s = qemu_mallocz(sizeof(RTCState));



    s->irq = irq;

    s->cmos_data[RTC_REG_A] = 0x26;

    s->cmos_data[RTC_REG_B] = 0x02;

    s->cmos_data[RTC_REG_C] = 0x00;

    s->cmos_data[RTC_REG_D] = 0x80;



    s->base_year = base_year;

    rtc_set_date_from_host(s);



    s->periodic_timer = qemu_new_timer(rtc_clock, rtc_periodic_timer, s);

    s->second_timer = qemu_new_timer(rtc_clock, rtc_update_second, s);

    s->second_timer2 = qemu_new_timer(rtc_clock, rtc_update_second2, s);



    s->next_second_time =

        qemu_get_clock(rtc_clock) + (get_ticks_per_sec() * 99) / 100;

    qemu_mod_timer(s->second_timer2, s->next_second_time);



    io_memory = cpu_register_io_memory(rtc_mm_read, rtc_mm_write, s);

    cpu_register_physical_memory(base, 2 << it_shift, io_memory);



    register_savevm("mc146818rtc", base, 1, rtc_save, rtc_load, s);

#ifdef TARGET_I386

    if (rtc_td_hack)

        register_savevm("mc146818rtc-td", base, 1, rtc_save_td, rtc_load_td, s);

#endif

    qemu_register_reset(rtc_reset, s);

    return s;

}
