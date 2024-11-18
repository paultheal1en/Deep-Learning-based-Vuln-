e1000e_intrmgr_delay_rx_causes(E1000ECore *core, uint32_t *causes)

{

    uint32_t delayable_causes;

    uint32_t rdtr = core->mac[RDTR];

    uint32_t radv = core->mac[RADV];

    uint32_t raid = core->mac[RAID];



    if (msix_enabled(core->owner)) {

        return false;

    }



    delayable_causes = E1000_ICR_RXQ0 |

                       E1000_ICR_RXQ1 |

                       E1000_ICR_RXT0;



    if (!(core->mac[RFCTL] & E1000_RFCTL_ACK_DIS)) {

        delayable_causes |= E1000_ICR_ACK;

    }



    /* Clean up all causes that may be delayed */

    core->delayed_causes |= *causes & delayable_causes;

    *causes &= ~delayable_causes;



    /* Check if delayed RX interrupts disabled by client

       or if there are causes that cannot be delayed */

    if ((rdtr == 0) || (causes != 0)) {

        return false;

    }



    /* Check if delayed RX ACK interrupts disabled by client

       and there is an ACK packet received */

    if ((raid == 0) && (core->delayed_causes & E1000_ICR_ACK)) {

        return false;

    }



    /* All causes delayed */

    e1000e_intrmgr_rearm_timer(&core->rdtr);



    if (!core->radv.running && (radv != 0)) {

        e1000e_intrmgr_rearm_timer(&core->radv);

    }



    if (!core->raid.running && (core->delayed_causes & E1000_ICR_ACK)) {

        e1000e_intrmgr_rearm_timer(&core->raid);

    }



    return true;

}
