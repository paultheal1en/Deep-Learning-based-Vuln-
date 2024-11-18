static void pm_update_sci(PIIX4PMState *s)

{

    int sci_level, pmsts;



    pmsts = acpi_pm1_evt_get_sts(&s->ar);

    sci_level = (((pmsts & s->ar.pm1.evt.en) &

                  (ACPI_BITMASK_RT_CLOCK_ENABLE |

                   ACPI_BITMASK_POWER_BUTTON_ENABLE |

                   ACPI_BITMASK_GLOBAL_LOCK_ENABLE |

                   ACPI_BITMASK_TIMER_ENABLE)) != 0) ||

        (((s->ar.gpe.sts[0] & s->ar.gpe.en[0]) &

          (PIIX4_PCI_HOTPLUG_STATUS | PIIX4_CPU_HOTPLUG_STATUS)) != 0);



    qemu_set_irq(s->irq, sci_level);

    /* schedule a timer interruption if needed */

    acpi_pm_tmr_update(&s->ar, (s->ar.pm1.evt.en & ACPI_BITMASK_TIMER_ENABLE) &&

                       !(pmsts & ACPI_BITMASK_TIMER_STATUS));

}
