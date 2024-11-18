int css_do_xsch(SubchDev *sch)

{

    SCSW *s = &sch->curr_status.scsw;

    PMCW *p = &sch->curr_status.pmcw;

    int ret;



    if (~(p->flags) & (PMCW_FLAGS_MASK_DNV | PMCW_FLAGS_MASK_ENA)) {

        ret = -ENODEV;

        goto out;

    }



    if (!(s->ctrl & SCSW_CTRL_MASK_FCTL) ||

        ((s->ctrl & SCSW_CTRL_MASK_FCTL) != SCSW_FCTL_START_FUNC) ||

        (!(s->ctrl &

           (SCSW_ACTL_RESUME_PEND | SCSW_ACTL_START_PEND | SCSW_ACTL_SUSP))) ||

        (s->ctrl & SCSW_ACTL_SUBCH_ACTIVE)) {

        ret = -EINPROGRESS;

        goto out;

    }



    if (s->ctrl & SCSW_CTRL_MASK_STCTL) {

        ret = -EBUSY;

        goto out;

    }



    /* Cancel the current operation. */

    s->ctrl &= ~(SCSW_FCTL_START_FUNC |

                 SCSW_ACTL_RESUME_PEND |

                 SCSW_ACTL_START_PEND |

                 SCSW_ACTL_SUSP);

    sch->channel_prog = 0x0;

    sch->last_cmd_valid = false;

    s->dstat = 0;

    s->cstat = 0;

    ret = 0;



out:

    return ret;

}
