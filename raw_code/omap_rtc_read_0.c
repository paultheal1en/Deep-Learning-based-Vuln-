static uint64_t omap_rtc_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_rtc_s *s = (struct omap_rtc_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;

    uint8_t i;



    if (size != 1) {

        return omap_badwidth_read8(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* SECONDS_REG */

        return to_bcd(s->current_tm.tm_sec);



    case 0x04:	/* MINUTES_REG */

        return to_bcd(s->current_tm.tm_min);



    case 0x08:	/* HOURS_REG */

        if (s->pm_am)

            return ((s->current_tm.tm_hour > 11) << 7) |

                    to_bcd(((s->current_tm.tm_hour - 1) % 12) + 1);

        else

            return to_bcd(s->current_tm.tm_hour);



    case 0x0c:	/* DAYS_REG */

        return to_bcd(s->current_tm.tm_mday);



    case 0x10:	/* MONTHS_REG */

        return to_bcd(s->current_tm.tm_mon + 1);



    case 0x14:	/* YEARS_REG */

        return to_bcd(s->current_tm.tm_year % 100);



    case 0x18:	/* WEEK_REG */

        return s->current_tm.tm_wday;



    case 0x20:	/* ALARM_SECONDS_REG */

        return to_bcd(s->alarm_tm.tm_sec);



    case 0x24:	/* ALARM_MINUTES_REG */

        return to_bcd(s->alarm_tm.tm_min);



    case 0x28:	/* ALARM_HOURS_REG */

        if (s->pm_am)

            return ((s->alarm_tm.tm_hour > 11) << 7) |

                    to_bcd(((s->alarm_tm.tm_hour - 1) % 12) + 1);

        else

            return to_bcd(s->alarm_tm.tm_hour);



    case 0x2c:	/* ALARM_DAYS_REG */

        return to_bcd(s->alarm_tm.tm_mday);



    case 0x30:	/* ALARM_MONTHS_REG */

        return to_bcd(s->alarm_tm.tm_mon + 1);



    case 0x34:	/* ALARM_YEARS_REG */

        return to_bcd(s->alarm_tm.tm_year % 100);



    case 0x40:	/* RTC_CTRL_REG */

        return (s->pm_am << 3) | (s->auto_comp << 2) |

                (s->round << 1) | s->running;



    case 0x44:	/* RTC_STATUS_REG */

        i = s->status;

        s->status &= ~0x3d;

        return i;



    case 0x48:	/* RTC_INTERRUPTS_REG */

        return s->interrupts;



    case 0x4c:	/* RTC_COMP_LSB_REG */

        return ((uint16_t) s->comp_reg) & 0xff;



    case 0x50:	/* RTC_COMP_MSB_REG */

        return ((uint16_t) s->comp_reg) >> 8;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
